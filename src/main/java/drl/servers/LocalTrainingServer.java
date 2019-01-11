package drl.servers;

import drl.AgentDependencyGraph;
import drl.MetaDecisionAgent;
import drl.agents.IAgent;
import drl.agents.MeleeButtonAgent;
import drl.agents.MeleeJoystickAgent;
import drl.collections.IReplayer;
import drl.collections.RandomReplayer;
import drl.collections.RankReplayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import static org.nd4j.linalg.ops.transforms.Transforms.abs;

public class LocalTrainingServer implements ITrainingServer{
    public static final int port = 1612;
    public static final long iterationsToTrain = 10000000;

    private HashMap<GraphMetadata, ComputationGraph> graphs;
    private HashMap<GraphMetadata, ComputationGraph>  targetGraphs;
    private AgentDependencyGraph dependencyGraph;
    private String[] outputs;
    private int inputSize;

    private IReplayer<DataPoint> dataPoints;

    private int batchSize;
    private Random random;
    private boolean connectFromNetwork;
    private int pointsGathered;
    private int iterations;
    private int pointWait;
    private boolean run;
    private boolean paused;

    private HashMap<Long, Thread> threads;

    public LocalTrainingServer(boolean connectFromNetwork, int maxReplaySize, int batchSize, AgentDependencyGraph dependencyGraph){
        this.batchSize = batchSize;
        this.connectFromNetwork = connectFromNetwork;

        this.pointWait = 5;

        this.dataPoints = new RankReplayer<>(maxReplaySize);
        this.random = new Random(324);

        this.dependencyGraph = dependencyGraph;

        this.run = true;
        this.paused = false;
        this.threads = new HashMap<>();

        this.graphs = new HashMap<>();
        this.targetGraphs = new HashMap<>();
        this.inputSize = MetaDecisionAgent.size;
    }

    public static void main(String[] args) throws Exception{
        Nd4j.getMemoryManager().togglePeriodicGc(false);

        AgentDependencyGraph dependencyGraph = new AgentDependencyGraph();

        IAgent joystickAgent = new MeleeJoystickAgent("M");
        IAgent cstickAgent = new MeleeJoystickAgent("C");
        IAgent abuttonAgent = new MeleeButtonAgent("A");
        dependencyGraph.addAgent(null, joystickAgent, "M");
        //dependencyGraph.addAgent(new String[]{"M"}, cstickAgent, "C");
        //dependencyGraph.addAgent(new String[]{"M"}, abuttonAgent, "A");

        int replaySize = Integer.parseInt(args[0]);
        int batchSize = Integer.parseInt(args[1]);
        LocalTrainingServer server = new LocalTrainingServer(true, replaySize, batchSize, dependencyGraph);

        InputStream input = new FileInputStream(args[2]);
        Scanner kb = new Scanner(input);
        while(kb.hasNext()){
            String line = kb.nextLine();
            String[] params = line.split(" ");

            float decayRate = Float.parseFloat(params[0]);
            int targetRotation = Integer.parseInt(params[1]);

            GraphMetadata metaData = new GraphMetadata(replaySize, batchSize, decayRate, targetRotation);

            File pretrained = new File(server.getModelName(metaData));

            if(pretrained.exists()){
                System.out.println("Loading model from file");
                ComputationGraph model = ModelSerializer.restoreComputationGraph(pretrained, true);
                server.addGraph(metaData, model);
                System.out.println(model.summary());
            }
            else{
                MetaDecisionAgent agent = new MetaDecisionAgent(dependencyGraph, 0);
                server.addGraph(metaData, agent.getMetaGraph());
                dependencyGraph.resetNodes();
                System.out.println(agent.getMetaGraph().summary());
            }
        }

        MetaDecisionAgent outputsAgent = new MetaDecisionAgent(dependencyGraph, 0);
        server.outputs = outputsAgent.getOutputNames();

        Thread t = new Thread(server);
        t.setPriority(Thread.MAX_PRIORITY);
        t.start();

        Queue<Integer> availablePorts = new LinkedList<>();

        Nd4j.getMemoryManager().invokeGc();

        for(int i = 1; i < 100; i++){
            availablePorts.add(LocalTrainingServer.port + i);
        }

        System.out.println("Accepting Clients");
        ServerSocket ss = new ServerSocket(LocalTrainingServer.port);
        while(server.run) {
            synchronized (server.threads) {
                LinkedList<Long> badThreads = new LinkedList<>();
                for (long creationTime : server.threads.keySet()) {
                    if (System.currentTimeMillis() - creationTime > 600000) {
                        Thread badThread = server.threads.get(creationTime);
                        if(badThread.isAlive()) {
                            badThread.interrupt();
                        }
                        badThreads.add(creationTime);
                    }
                }
                for(long creationTime : badThreads) {
                    server.threads.remove(creationTime);
                }
            }

            Socket mainSocket = ss.accept();
            while(availablePorts.isEmpty()) {
                Thread.sleep(100);
            }
            int port = availablePorts.poll();

            ObjectOutputStream mainStream = new ObjectOutputStream(mainSocket.getOutputStream());
            mainStream.writeObject(port);
            mainSocket.close();

            ServerSocket connectionServer = new ServerSocket(port);
            Socket socket = connectionServer.accept();
            Runnable r = new Runnable() {
                @Override
                public void run() {
                    try {

                        socket.setSoTimeout(600000);
                        System.out.println("Client connected on port " + port);

                        OutputStream rawOutput = socket.getOutputStream();
                        ObjectOutputStream output = new ObjectOutputStream(rawOutput);
                        ObjectInputStream input = new ObjectInputStream(socket.getInputStream());

                        try {
                            while (true) {
                                String message = (String) input.readObject();
                                switch (message) {
                                    case ("addData"):
                                        try {
                                            INDArray[] startState = (INDArray[]) input.readObject();
                                            INDArray[] endState = (INDArray[]) input.readObject();
                                            INDArray[] masks = (INDArray[]) input.readObject();
                                            float score = (float) input.readObject();

                                            if (server.dataPoints.size() % 10000 == 100) {
                                                server.writeStateToImage(startState, "start");
                                            }

                                            server.addData(startState, endState, masks, score);
                                        } catch (Exception e) {
                                            System.out.println("Error while attempting to upload a data point, point destroyed");
                                        }
                                        break;
                                    case ("getUpdatedNetwork"):
                                        Iterator<GraphMetadata> iterator = server.graphs.keySet().iterator();
                                        GraphMetadata randomData = iterator.next();
                                        for (int i = 1; i < server.random.nextInt(server.graphs.size()); i++) {
                                            randomData = iterator.next();
                                        }

                                        Path modelPath = Paths.get(server.getModelName(randomData));
                                        byte[] modelBytes = Files.readAllBytes(modelPath);
                                        output.writeObject(modelBytes);
                                        break;
                                    case ("getDependencyGraph"):
                                        output.writeObject(server.getDependencyGraph());
                                        break;
                                    case ("getProb"):
                                        double prob = server.getProb();
                                        output.writeObject(prob);
                                        break;
                                    default:
                                        System.out.println("Got unregistered input, exiting because of " + message);
                                        throw new InvalidObjectException("Improper network request");
                                }
                            }
                        } catch (Exception e) {
                            e.printStackTrace(System.out);
                            System.out.println("Client disconnected from port " + port);
                        } finally {
                            System.out.println("Closing Server Socket");
                            socket.close();
                            connectionServer.close();
                            Nd4j.getMemoryManager().invokeGc();
                        }
                    } catch (IOException e) {
                        e.printStackTrace(System.out);
                        System.out.println("Error opening port " + port);
                    }
                    availablePorts.add(port);
                }
            };

            long creationTime = System.currentTimeMillis();
            Thread sockThread = new Thread(r);
            sockThread.start();
            synchronized (server.threads) {
                server.threads.put(creationTime, sockThread);
            }
        }
    }

    @Override
    public void run() {
        long batchTime = 0;
        long concatTime = 0;
        long buildTime = 0;
        long fitTime = 0;

        double alpha = .6;
        double probabilitySum = this.getProbabilitySum(alpha, this.dataPoints.getMaxSize());
        ArrayList<Integer> probabilityIndexes = this.getProbabilityIntervals(this.batchSize, alpha, this.dataPoints.getMaxSize());
        INDArray[] blankInput = new INDArray[]{Nd4j.ones(1, 4, 84, 84)};
        INDArray[] blankLabels = new INDArray[this.outputs.length];
        for(int i = 0; i < blankLabels.length; i++){
            blankLabels[i] = Nd4j.ones(1);
        }
        //this.dataPoints.prepopulate(new DataPoint(blankInput, blankInput, blankLabels, blankLabels));

        while(this.run){
            System.out.print("");

            boolean sufficientDataGathered = this.pointsGathered > this.dataPoints.getMaxSize();

            if (!paused && sufficientDataGathered && iterations <= pointsGathered) {
                long startTime = System.currentTimeMillis();

                DataPoint[] batchPoints = new DataPoint[this.batchSize];
                INDArray[][] startStates = new INDArray[this.batchSize][];
                INDArray[][] endStates = new INDArray[this.batchSize][];
                INDArray[][] labels = new INDArray[this.batchSize][];
                INDArray[][] masks = new INDArray[this.batchSize][];

                double beta = .4 + (this.getProb()) * .6;
                double minP = Math.pow(1.0 / ((double) (this.dataPoints.getMaxSize())), alpha) / probabilitySum;
                double maxW = Math.pow(this.dataPoints.getMaxSize() * minP, -1.0 * beta);
                double[] wArray = new double[this.batchSize];

                synchronized (this.dataPoints) {
                    int lastLabel = 0;
                    for (int i = 0; i < probabilityIndexes.size(); i++) {
                        int index = probabilityIndexes.get(i) == 0 ? 0 : this.random.nextInt(probabilityIndexes.get(i) - lastLabel) + lastLabel + 1;
                        DataPoint data = this.dataPoints.get(index - i);
                        batchPoints[i] = data;

                        double p = Math.pow(1.0 / ((double) (index + 1)), alpha) / probabilitySum;
                        wArray[i] = Math.pow(this.dataPoints.getMaxSize() * p, -1.0 * beta) / maxW;

                        lastLabel = probabilityIndexes.get(i);

                        startStates[i] = data.getStartState();
                        endStates[i] = data.getEndState();
                        labels[i] = data.getLabels();
                        masks[i] = data.getMasks();
                    }
                }

                INDArray w = Nd4j.create(wArray, new int[] {wArray.length, 1}, 'c');

                long batch = System.currentTimeMillis();

                DataPoint cumulativeData = new DataPoint(this.concatSet(startStates), this.concatSet(endStates), this.concatSet(labels), this.concatSet(masks));

                long concat = System.currentTimeMillis();

                for (GraphMetadata metaData : this.graphs.keySet()) {
                    long graphStart = System.currentTimeMillis();

                    ComputationGraph graph = this.graphs.get(metaData);
                    ComputationGraph targetGraph = this.targetGraphs.get(metaData);

                    INDArray[] curLabels = graph.output(cumulativeData.getEndState());
                    INDArray[] targetLabels = targetGraph.output(cumulativeData.getEndState());

                    INDArray[] targetMaxs = new INDArray[curLabels.length];

                    for (ArrayList<Integer> inds : this.dependencyGraph.getAgentInds(this.outputs)) {
                        int concatInd = 0;
                        INDArray[] curIndLabels = new INDArray[inds.size()];
                        for (int i : inds) {
                            curIndLabels[concatInd] = curLabels[i];
                            concatInd++;
                        }
                        INDArray max = Nd4j.concat(1, curIndLabels);
                        max = Nd4j.max(max, 1);

                        INDArray[] maxBools = new INDArray[curLabels.length];
                        for (int i : inds) {
                            maxBools[i] = curLabels[i].eq(max);
                        }

                        INDArray targetMax = Nd4j.zeros(targetLabels[0].shape());
                        for (int i : inds) {
                            targetMax = targetMax.add(maxBools[i].mul(targetLabels[i]));
                        }

                        for (int i : inds) {
                            targetMaxs[i] = targetMax;
                        }
                    }

                    MultiDataSet dataSet = cumulativeData.getDataSetWithQOffset(targetMaxs, metaData.decayRate);

                    INDArray[] inputLabels = graph.output(dataSet.getFeatures());
                    INDArray[] error = new INDArray[dataSet.getLabels().length];
                    INDArray absTotalError = Nd4j.zeros(w.shape());
                    INDArray[] qLabels = dataSet.getLabels();
                    INDArray[] dataMasks = cumulativeData.getMasks();

                    for(int i = 0; i < error.length; i++){
                        error[i] = qLabels[i].sub(inputLabels[i]);
                        absTotalError = absTotalError.add(abs(error[i].mul(dataMasks[i])));
                    }

                    INDArray[] weightedLabels = new INDArray[dataSet.getLabels().length];

                    for(int i = 0; i < weightedLabels.length; i++){
                        weightedLabels[i] = inputLabels[i].add(error[i].mul(w));
                    }

                    dataSet.setLabels(weightedLabels);

                    double[] errors = absTotalError.toDoubleVector();
                    synchronized (this.dataPoints) {
                        for (int k = 0; k < batchPoints.length; k++) {
                            this.dataPoints.add(errors[k], batchPoints[k]);
                        }
                    }

                    long graphBuild = System.currentTimeMillis();

                    graph.fit(dataSet);

                    long graphFit = System.currentTimeMillis();

                    batchTime += batch - startTime;
                    concatTime += concat - batch;
                    buildTime += graphBuild - graphStart;
                    fitTime += graphFit - graphBuild;

                    if (metaData.targetRotation != 0 && iterations % metaData.targetRotation == 0) {
                        this.targetGraphs.put(metaData, this.getUpdatedNetwork(metaData, true));
                    }

                    if (iterations % 100 == 0) {
                        Nd4j.getMemoryManager().invokeGc();
                        System.out.println("Total batch time: " + batchTime + " average was " + (batchTime / 100));
                        System.out.println("Total concat time: " + concatTime + " average was " + (concatTime / 100));
                        System.out.println("Total build time: " + buildTime + " average was " + (buildTime / 100));
                        System.out.println("Total fit time: " + fitTime + " average was " + (fitTime / 100));
                        batchTime = 0;
                        concatTime = 0;
                        buildTime = 0;
                        fitTime = 0;
                    }
                }

                iterations++;

            }
            else {
                try {
                    Thread.sleep(10);
                }
                catch (Exception e){
                    System.out.println("This Thread is Weak");
                }
            }

            if (this.pointWait != 0) {
                this.pointWait -= 1;
            }

            int serverIterations = this.graphs.get(this.graphs.keySet().iterator().next()).getIterationCount();

            if(serverIterations >= LocalTrainingServer.iterationsToTrain){
                this.run = false;
            }
        }
    }

    @Override
    public void addData(INDArray[] startState, INDArray[] endState, INDArray[] masks, float score) {
        pointWait = 5;

        INDArray[] labels = new INDArray[masks.length];

        for (int i = 0; i < masks.length; i++) {
            labels[i] = masks[i].mul(score);
        }

        DataPoint data = new DataPoint(startState, endState, labels, masks);

        synchronized (this.dataPoints){
            this.dataPoints.add(100, data);
        }

        pointsGathered++;

        if(pointsGathered % 100 == 0){
            System.out.println(pointsGathered + " points gathered");
        }
    }

    @Override
    public ComputationGraph getUpdatedNetwork() {
        Iterator<GraphMetadata> iterator = graphs.keySet().iterator();
        GraphMetadata randomData = iterator.next();
        for(int i = 1; i < this.random.nextInt(graphs.size()); i++){
            randomData = iterator.next();
        }
        return this.getUpdatedNetwork(randomData, !this.connectFromNetwork);
    }

    private ComputationGraph getUpdatedNetwork(GraphMetadata metaData, boolean clone) {
        try{
            ComputationGraph graph = this.graphs.get(metaData);


            if(!clone){
                return graph;
            }
            else {
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                ModelSerializer.writeModel(graph, baos, true);

                byte[] modelBytes = baos.toByteArray();

                File f = new File(this.getModelName(metaData));

                if (f.exists()) {
                    f.delete();
                }

                FileOutputStream fout = new FileOutputStream(f);
                fout.write(modelBytes);

                ByteArrayInputStream bais = new ByteArrayInputStream(modelBytes);
                return ModelSerializer.restoreComputationGraph(bais, true);
            }
        }
        catch(Exception e){
            System.out.println(e);
        }
        return null;
    }

    @Override
    public AgentDependencyGraph getDependencyGraph() {
        return this.dependencyGraph;
    }

    @Override
    public void pause() {
        paused = true;
    }

    @Override
    public void resume() {
        paused = false;
    }

    @Override
    public void stop() {
        this.run  = false;
    }

    @Override
    public double getProb() {
        long iterations = this.graphs.get(this.graphs.keySet().iterator().next()).getIterationCount();
        double prob = (double) iterations / (double) LocalTrainingServer.iterationsToTrain;
        return prob;
    }

    protected String getModelName(GraphMetadata metaData){
        StringBuilder sb = new StringBuilder();
        sb.append("model-");
        //for(String output : this.agent.getOutputNames()){
        //    sb.append(output);
        //    sb.append("-");
        //}
        sb.append(metaData.getName());
        sb.append(".mod");
        return sb.toString();
    }

    protected void addGraph(GraphMetadata metaData, ComputationGraph graph){
        this.graphs.put(metaData, graph);
        this.targetGraphs.put(metaData, this.getUpdatedNetwork(metaData, metaData.targetRotation != 0));

        int listenerFrequency = 1;
        boolean reportScore = true;
        boolean reportGC = true;
        this.graphs.get(metaData).setListeners(new PerformanceListener(100, reportScore));
    }

    private INDArray[] concatSet(INDArray[][] set){
        INDArray[] result = new INDArray[set[0].length];

        for(int j = 0; j < result.length; j++) {
            INDArray[] toConcat = new INDArray[set.length];

            for (int i = 0; i < set.length; i++) {
                toConcat[i] = set[i][j];
            }

            result[j] = Nd4j.concat(0, toConcat);
        }

        return result;
    }

    private ArrayList<Integer> getProbabilityIntervals(int batchSize, double alpha, int totalSize){
        double probSum = this.getProbabilitySum(alpha, totalSize);

        ArrayList<Integer> intervals = new ArrayList<>();

        double equivalenceSize = 1.0 / batchSize;
        double tierSum = 0;
        for(int i = 0; i < totalSize; i++){
            tierSum += Math.pow(1.0 / ((double)(i + 1)), alpha) / probSum;
            if(tierSum >= equivalenceSize){
                intervals.add(i);
                tierSum = tierSum - equivalenceSize;
            }
        }

        if(intervals.size() != batchSize) {
            intervals.add(totalSize - 1);
        }

        return intervals;
    }

    private double getProbabilitySum(double alpha, int totalSize){
        double probSum = 0;
        for(int i = 0; i < totalSize; i++){
            probSum += Math.pow(1.0 / ((double)(i + 1)), alpha);
        }
        return probSum;
    }

    private void writeStateToImage(INDArray[] state, String fileName){
        try{
            File f = new File(fileName + ".jpg");
            BufferedImage img = new BufferedImage(this.inputSize, this.inputSize, BufferedImage.TYPE_3BYTE_BGR);

            INDArray stateImage = state[0].getColumn(3).mul(255);
            int[] pixelValues = Nd4j.toFlattened(stateImage).toIntVector();

            int index = 0;
            for(int i = 0; i < this.inputSize; i++){
                for(int j = 0; j < this.inputSize; j++){
                    img.setRGB(j, i, pixelValues[index] * 0x10101);
                    index++;
                }
            }

            ImageIO.write(img, "jpg", f);
        }catch(IOException e){
            System.out.println(e);
        }
    }

}
