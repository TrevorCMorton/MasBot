package drl.servers;

import drl.AgentDependencyGraph;
import drl.MetaDecisionAgent;
import drl.WeightedActivationRelu;
import drl.agents.CombinationControlAgent;
import drl.agents.IAgent;
import drl.agents.MeleeButtonAgent;
import drl.agents.MeleeJoystickAgent;
import drl.collections.IReplayer;
import drl.collections.RandomReplayer;
import drl.collections.RankReplayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import static org.nd4j.linalg.ops.transforms.Transforms.abs;

public class LocalTrainingServer implements ITrainingServer{
    public static final int port = 1612;
    private boolean stats;
    private HashMap<Integer, Double> statsStorage;
    private HashMap<Integer, Double> timeStorage;
    private HashMap<Integer, Integer> gathered;
    double bestCount = Double.MAX_VALUE * -1.0;

    private ComputationGraph grap;
    private MetaDecisionAgent decisionAgent;
    private ComputationGraph  targetGrap;
    private GraphMetadata metadata;
    private AgentDependencyGraph dependencyGraph;
    private String[] outputs;
    private int inputSize;

    private IReplayer<DataPoint> dataPoints;

    private long iterationsToTrain;
    private int batchSize;
    private double learningRate;
    private boolean prioritizedReplay;
    private List<IActivation> activations;
    private Random random;
    private boolean connectFromNetwork;
    private int pointsGathered;
    private int iterations;
    private int pointWait;
    private boolean run;
    private boolean paused;
    private String latestFile;

    private HashMap<Long, Thread> threads;

    private int evalIters = 10;

    public LocalTrainingServer(boolean connectFromNetwork, long iterationsToTrain, int maxReplaySize, int batchSize, double learningRate, boolean prioritizedReplay, AgentDependencyGraph dependencyGraph){
        this.iterationsToTrain = iterationsToTrain;
        this.batchSize = batchSize;
        this.learningRate = learningRate;
        this.connectFromNetwork = connectFromNetwork;

        this.pointWait = 5;

        this.statsStorage = new HashMap<>();
        this.timeStorage = new HashMap<>();
        this.gathered = new HashMap<>();

        this.prioritizedReplay = prioritizedReplay;
        this.activations = new ArrayList<>();
        if(this.prioritizedReplay){
            this.dataPoints = new RankReplayer<>(maxReplaySize);
        }
        else{
            this.dataPoints = new RandomReplayer<>(maxReplaySize);
        }
        this.random = new Random(324);

        this.dependencyGraph = dependencyGraph;

        this.run = true;
        this.paused = false;
        this.threads = new HashMap<>();

        this.inputSize = MetaDecisionAgent.size;
    }

    public static void main(String[] args) throws Exception{
        Nd4j.getMemoryManager().togglePeriodicGc(false);
        Nd4j.setDataType(DataBuffer.Type.FLOAT);

        AgentDependencyGraph dependencyGraph = new AgentDependencyGraph();
        IAgent joystickAgent = new MeleeJoystickAgent("M");
        IAgent bbuttonAgent = new MeleeButtonAgent("B");
        IAgent cstickAgent = new MeleeJoystickAgent("C");
        IAgent abuttonAgent = new MeleeButtonAgent("A");
        IAgent combination = new CombinationControlAgent(new String[][]{{"MR", "MN", "MNE", "ME", "MSE", "MS", "MSW", "MW", "MNW" },{"PA", "RA"}});
        //dependencyGraph.addAgent(null, bbuttonAgent, "B");
        //dependencyGraph.addAgent(new String[]{"M"}, abuttonAgent, "A");
        //dependencyGraph.addAgent(new String[]{"B"}, joystickAgent, "M");
        //dependencyGraph.addAgent(new String[]{"M"}, cstickAgent, "C");
        dependencyGraph.addAgent(null, combination, "Comb");

        int replaySize = Integer.parseInt(args[0]);
        int batchSize = Integer.parseInt(args[1]);
        double learningRate = Double.parseDouble(args[2]);
        boolean prioritizedReplay = Boolean.parseBoolean(args[3]);
        long iterationsToTrain = Long.parseLong(args[4]);
        LocalTrainingServer server = new LocalTrainingServer(true, iterationsToTrain, replaySize, batchSize, learningRate, prioritizedReplay, dependencyGraph);

        InputStream input = new FileInputStream(args[5]);
        Scanner kb = new Scanner(input);
        String line = kb.nextLine();
        String[] params = line.split(" ");

        float decayRate = Float.parseFloat(params[0]);
        int targetRotation = Integer.parseInt(params[1]);
        boolean stats = Boolean.parseBoolean(params[2]);

        server.stats = stats;

        server.metadata = new GraphMetadata(replaySize, batchSize, decayRate, targetRotation);

        File pretrained = new File(server.getModelName());

        if(!prioritizedReplay && pretrained.exists()){
            System.out.println("Loading model from file");
            ComputationGraph model = ModelSerializer.restoreComputationGraph(pretrained, true);
            server.setGraph(model);
            System.out.println(model.summary());
        }
        else{
            MetaDecisionAgent agent = new MetaDecisionAgent(dependencyGraph, server.activations,0, learningRate, !prioritizedReplay);
            server.setGraph(agent.getMetaGraph());
            dependencyGraph.resetNodes();
            System.out.println(agent.getMetaGraph().summary());
        }
        server.setTargetGraph(server.grap);
        server.decisionAgent = new MetaDecisionAgent(dependencyGraph, server.activations, 1, learningRate, !prioritizedReplay);
        server.outputs = server.decisionAgent.getOutputNames();

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
                        boolean stats = server.isStatsRunner();
                        int iterations = server.iterations;
                        int next = iterations + server.metadata.targetRotation;
                        byte[] modelBytes = null;
                        int pointsGathered = 0;
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
                                                INDArray endState = (INDArray) input.readObject();
                                                INDArray[] masks = (INDArray[]) input.readObject();
                                                float score = (float) input.readObject();
                                                INDArray[] startLabels = (INDArray[]) input.readObject();
                                                INDArray[] endLabels = (INDArray[]) input.readObject();

                                                if (server.dataPoints.size() % 10000 == 100) {
                                                    //server.writeStateToImage(startState, "start");
                                                }

                                                if(!stats) {
                                                    server.addData(startState, endState, masks, score, startLabels, endLabels);
                                                }
                                                pointsGathered++;
                                            } catch (Exception e) {
                                                System.out.println("Error while attempting to upload a data point, point destroyed");
                                                System.out.println(e);
                                                e.printStackTrace();
                                            }
                                        break;
                                    case ("addScore"):
                                        double score = (double) input.readObject();
                                        if(stats){
                                            server.addScore(score);
                                            if(server.timeStorage.containsKey(iterations)){
                                                pointsGathered += server.timeStorage.get(iterations);
                                            }
                                            server.timeStorage.put(iterations, (double)pointsGathered);

                                            synchronized (server) {
                                                if (score > server.bestCount && modelBytes != null) {
                                                    server.bestCount = score;

                                                    File f = new File("models/best.mod");
                                                    FileOutputStream fout = new FileOutputStream(f);
                                                    fout.write(modelBytes);
                                                }
                                            }

                                            if(!server.gathered.containsKey(iterations)){
                                                server.gathered.put(iterations, 0);
                                            }

                                            server.gathered.put(iterations, server.gathered.get(iterations) + 1);

                                            if(server.gathered.get(iterations) >= server.evalIters){
                                                server.iterations = next;
                                            }
                                        }
                                        break;
                                    case ("getUpdatedNetwork"):
                                        try{
                                            Path modelPath;
                                            if(stats){
                                                modelPath = Paths.get(server.getModelName());
                                                System.out.println("Sending network" + server.getModelName());

                                            }
                                            else {
                                                modelPath = Paths.get(server.latestFile);
                                            }
                                            modelBytes = Files.readAllBytes(modelPath);
                                            output.writeObject(modelBytes);
                                        }
                                        catch (NoSuchFileException f){
                                            System.out.println("Finishing evaluation");
                                            server.run = false;
                                            return;
                                        }
                                        break;
                                    case ("getDependencyGraph"):
                                        output.writeObject(server.getDependencyGraph());
                                        break;
                                    case ("getProb"):
                                        double prob;
                                        if(stats){
                                            prob = 1.0;
                                        }
                                        else{
                                            prob = server.getProb();
                                        }
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

        double alpha = .7;
        double probabilitySum = this.getProbabilitySum(alpha, this.dataPoints.getMaxSize());
        ArrayList<Integer> probabilityIndexes = this.getProbabilityIntervals(this.batchSize, alpha, this.dataPoints.getMaxSize());
        /*
        this.decisionAgent.setMetaGraph(this.graphs.get(this.graphs.keySet().iterator().next()));
        INDArray[] randInput = this.decisionAgent.getState(Nd4j.rand(new int[]{1, MetaDecisionAgent.depth, MetaDecisionAgent.size, MetaDecisionAgent.size}));
        INDArray[] blankLabels = new INDArray[this.outputs.length];
        for(int i = 0; i < blankLabels.length; i++){
            blankLabels[i] = Nd4j.ones(1);
        }
        this.dataPoints.prepopulate(new DataPoint(randInput, Nd4j.rand(new int[]{1, MetaDecisionAgent.depth, MetaDecisionAgent.size, MetaDecisionAgent.size}), blankLabels, blankLabels));
        */
        while(this.run || this.iterations < this.iterationsToTrain && this.stats){
            System.out.print("");

            boolean sufficientDataGathered = this.pointsGathered > this.dataPoints.getMaxSize() && this.pointWait > 0;

            if (!paused && sufficientDataGathered && iterations <= pointsGathered) {
                long startTime = System.currentTimeMillis();

                DataPoint[] batchPoints = new DataPoint[this.batchSize];
                INDArray[][] startStates = new INDArray[this.batchSize][];
                INDArray[] endStates = new INDArray[this.batchSize];
                INDArray[][] labels = new INDArray[this.batchSize][];
                INDArray[][] masks = new INDArray[this.batchSize][];

                double beta = .5 + (this.getProb()) * .5;
                double minP = Math.pow(1.0 / ((double) (this.dataPoints.getMaxSize())), alpha) / probabilitySum;
                double maxW = Math.pow(this.dataPoints.getMaxSize() * minP, -1.0 * beta);
                double[] wArray = new double[this.batchSize];

                synchronized (this.dataPoints) {
                    if(prioritizedReplay) {
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
                    else{
                        for(int i = 0; i < this.batchSize; i++){
                            int ind = (int)(Math.random() * this.dataPoints.size());
                            DataPoint data = this.dataPoints.get(ind);
                            batchPoints[i] = data;
                            startStates[i] = data.getStartState();
                            endStates[i] = data.getEndState();
                            labels[i] = data.getLabels();
                            masks[i] = data.getMasks();
                        }
                    }
                }

                INDArray w = Nd4j.create(wArray, new int[] {wArray.length, 1}, 'c');

                long batch = System.currentTimeMillis();

                INDArray[] cumStart = this.concatSet(startStates);
                INDArray cumEnd = Nd4j.concat(0, endStates);
                INDArray[] cumLabels = this.concatSet(labels);
                INDArray[] cumMasks = this.concatSet(masks);

                DataPoint cumulativeData = new DataPoint(cumStart, cumEnd, cumLabels, cumMasks);

                long concat = System.currentTimeMillis();

                long graphStart = System.currentTimeMillis();

                this.decisionAgent.setMetaGraph(this.grap);
                INDArray[] endState = this.decisionAgent.getState(cumulativeData.endState);

                INDArray[] curLabels = this.grap.output(endState);
                INDArray[] targetLabels = this.targetGrap.output(endState);

                INDArray[] targetMaxs = new INDArray[curLabels.length];

                ArrayList<INDArray> maxs = new ArrayList<>();

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
                    //maxs.add(targetMax);

                    for (int i : inds) {
                        targetMaxs[i] = targetMax;
                    }

                }
                    /*
                    INDArray maxSum = Nd4j.zeros(maxs.get(0).shape());
                    for(int i = 0; i < maxs.size(); i++){
                        maxSum.add(maxs.get(i));
                    }
                    maxSum.div(maxs.size());

                    for(int i = 0; i < targetMaxs.length; i++){
                        targetMaxs[i] = maxSum;
                    }
                    */
                MultiDataSet dataSet = cumulativeData.getDataSetWithQOffset(targetMaxs, this.metadata.decayRate);

                long graphBuild = System.currentTimeMillis();

                if(this.prioritizedReplay) {

                    INDArray[] inputLabels = this.grap.output(dataSet.getFeatures());
                    INDArray[] qLabels = dataSet.getLabels();
                    INDArray[] dataMasks = cumulativeData.getMasks();
                    INDArray[] squaredError = new INDArray[dataSet.getLabels().length];
                    INDArray absTotalError = Nd4j.zeros(w.shape());

                    for (int i = 0; i < squaredError.length; i++) {
                        INDArray error = qLabels[i].sub(inputLabels[i]).mul(dataMasks[i]);
                        squaredError[i] = error.mul(error);
                        absTotalError = absTotalError.add(abs(error));
                    }

                    double[] errors = absTotalError.toDoubleVector();
                    synchronized (this.dataPoints) {
                        for (int k = 0; k < batchPoints.length; k++) {
                            this.dataPoints.add(errors[k], batchPoints[k]);
                        }
                    }

                    INDArray params = this.grap.params().dup();

                    for(IActivation activation : this.activations){
                        ((WeightedActivationRelu)activation).setWeight(w);
                    }

                    this.grap.feedForward(dataSet.getFeatures(), true, false);
                    Gradient grad = this.grap.backpropGradient(squaredError);
                    this.grap.getUpdater().update(grad, iterations, 0, this.batchSize, LayerWorkspaceMgr.noWorkspaces());
                    INDArray newParams = params.sub(grad.gradient());
                    this.grap.setParams(newParams);

                }
                else {
                    this.grap.fit(dataSet);
                }
                long graphFit = System.currentTimeMillis();

                batchTime += batch - startTime;
                concatTime += concat - batch;
                buildTime += graphBuild - graphStart;
                fitTime += graphFit - graphBuild;

                if (this.metadata.targetRotation != 0 && this.iterations % this.metadata.targetRotation == 0) {
                    this.setTargetGraph(this.grap);

                    if(this.iterations + 1 >= this.iterationsToTrain){
                        this.run = false;
                    }
                }

                if (iterations % 50 == 0) {
                    Nd4j.getMemoryManager().invokeGc();
                    System.out.println("Finished iteration " + this.iterations);
                    System.out.println("Total batch time: " + batchTime + " average was " + (batchTime / 50));
                    System.out.println("Total concat time: " + concatTime + " average was " + (concatTime / 50));
                    System.out.println("Total build time: " + buildTime + " average was " + (buildTime / 50));
                    System.out.println("Total fit time: " + fitTime + " average was " + (fitTime / 50));
                    batchTime = 0;
                    concatTime = 0;
                    buildTime = 0;
                    fitTime = 0;
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
        }

        try {
            writeHashMapToCsv("scores", statsStorage);
            writeHashMapToCsv("times", timeStorage);
        }
        catch (Exception e){
            System.out.println("Failed writing scores to file " + e);
        }
    }

    @Override
    public void addData(INDArray[] startState, INDArray endState, INDArray[] masks, float score, INDArray[] startLabels, INDArray[] endLabels) {
        pointWait = 5;

        INDArray[] labels = new INDArray[masks.length];

        for (int i = 0; i < masks.length; i++) {
            labels[i] = masks[i].mul(score);
        }

        DataPoint data = new DataPoint(startState, endState, labels, masks);

        INDArray absTotalError = Nd4j.zeros(startLabels[0].shape());

        for(int i = 0; i < startLabels.length; i++){
            INDArray error = endLabels[i].sub(startLabels[i]);
            absTotalError = absTotalError.add(abs(error.mul(masks[i])));
        }

        double[] errors = absTotalError.toDoubleVector();

        synchronized (this.dataPoints){
            this.dataPoints.add(errors[0], data);
        }

        pointsGathered++;

        if(pointsGathered % 100 == 0){
            System.out.println(pointsGathered + " points gathered");
        }
    }

    @Override
    public void addScore(double score) {
        if(this.statsStorage.containsKey(this.iterations)){
            score += this.statsStorage.get(this.iterations);
        }
        this.statsStorage.put(this.iterations, score);
    }

    @Override
    public ComputationGraph getUpdatedNetwork() {
        return this.getUpdatedNetwork(!this.connectFromNetwork);
    }

    private ComputationGraph getUpdatedNetwork(boolean clone) {
        try{
            ComputationGraph graph = this.grap;


            if(!clone){
                return graph;
            }
            else {
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                ModelSerializer.writeModel(graph, baos, true);

                byte[] modelBytes = baos.toByteArray();

                String fileName = this.getModelName();
                File f = new File(fileName);

                if (f.exists()) {
                    f.delete();
                }

                File dir = new File("./models");
                dir.mkdir();

                FileOutputStream fout = new FileOutputStream(f);
                fout.write(modelBytes);

                this.latestFile = fileName;

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
        double prob = (double) this.iterations / (double) this.iterationsToTrain * .9;
        return prob;
    }

    protected String getModelName(){
        return "./models/" + this.iterations + ".mod";
    }

    protected void setGraph(ComputationGraph graph){
        this.grap = graph;
        this.grap.setListeners(new PerformanceListener(100, true));
    }

    protected void setTargetGraph(ComputationGraph graph){
        this.targetGrap = this.getUpdatedNetwork(true);
    }

    private synchronized boolean isStatsRunner(){
        return this.stats;
    }

    private void writeHashMapToCsv(String fileName, HashMap<Integer, Double> dataMap ) throws Exception {
        File f = new File(fileName + ".csv");
        FileOutputStream fout = new FileOutputStream(f);
        String csvHeader = "iterations," + fileName + "\n";
        fout.write(csvHeader.getBytes());
        for(int key : dataMap.keySet()){
            String csvLine = key + "," + (dataMap.get(key) / (double)this.gathered.get(key)) + "\n";
            fout.write(csvLine.getBytes());
        }
        fout.close();
    }

    private INDArray[] concatSet(INDArray[][] set){
        INDArray[] result = new INDArray[set[0].length];

        for(int j = 0; j < result.length; j++) {
            INDArray[] toConcat = new INDArray[set.length];

            for (int i = 0; i < set.length; i++) {
                toConcat[i] = set[i][j];
            }

            try {
                result[j] = Nd4j.concat(0, toConcat);
            }
            catch(Exception e){
                for(INDArray arr : toConcat){
                    System.out.println(Arrays.toString(arr.shape()));
                }
            }
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

            INDArray stateImage = state[0].getColumn(MetaDecisionAgent.depth - 1).mul(255);
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

    private INDArray[] getArraySlices(INDArray[] arrays, int ind){
        INDArray[] slices = new INDArray[arrays.length];

        for(int i = 0; i < arrays.length; i++){
            long[] shape = arrays[0].shape().clone();
            shape[0] = 1;
            slices[i] = arrays[i].slice(ind, 0);
            slices[i] = slices[i].reshape(shape);
        }

        return slices;
    }

}
