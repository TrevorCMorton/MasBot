package drl.servers;

import drl.AgentDependencyGraph;
import drl.MetaDecisionAgent;
import drl.agents.IAgent;
import drl.agents.MeleeButtonAgent;
import drl.agents.MeleeJoystickAgent;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.*;

import static org.nd4j.linalg.ops.transforms.Transforms.abs;

public class LocalTrainingServer implements ITrainingServer{
    public static final int[] ports = { 1612, 1613, 1614, 1615, 1616 };

    private HashMap<GraphMetadata, ComputationGraph> graphs;
    private HashMap<GraphMetadata, ComputationGraph>  targetGraphs;
    private AgentDependencyGraph dependencyGraph;
    private String[] outputs;
    private int inputSize;

    private CircularFifoQueue<DataPoint> dataPoints;

    private int batchSize;
    private Random random;
    private boolean connectFromNetwork;
    private int pointsGathered;
    private int iterations;
    private int pointWait;
    private boolean run;

    public LocalTrainingServer(boolean connectFromNetwork, int maxReplaySize, int batchSize, AgentDependencyGraph dependencyGraph){
        this.batchSize = batchSize;
        this.connectFromNetwork = connectFromNetwork;

        this.pointWait = 5;

        this.dataPoints = new CircularFifoQueue<>(maxReplaySize);
        this.random = new Random(324);

        this.dependencyGraph = dependencyGraph;

        this.run = true;

        this.graphs = new HashMap<>();
        this.targetGraphs = new HashMap<>();
        this.inputSize = MetaDecisionAgent.size;
    }

    public static void main(String[] args) throws Exception{
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
            int commDepth = Integer.parseInt(params[1]);
            int targetRotation = Integer.parseInt(params[2]);

            GraphMetadata metaData = new GraphMetadata(replaySize, batchSize, decayRate, commDepth, targetRotation);

            File pretrained = new File(server.getModelName(metaData));

            if(pretrained.exists()){
                System.out.println("Loading model from file");
                ComputationGraph model = ModelSerializer.restoreComputationGraph(pretrained, true);
                server.addGraph(metaData, model);
                System.out.println(model.summary());
            }
            else{
                MetaDecisionAgent agent = new MetaDecisionAgent(dependencyGraph, 0, commDepth);
                server.addGraph(metaData, agent.getMetaGraph());
                dependencyGraph.resetNodes();
                System.out.println(agent.getMetaGraph().summary());
            }
        }

        MetaDecisionAgent outputsAgent = new MetaDecisionAgent(dependencyGraph, 0, 3);
        server.outputs = outputsAgent.getOutputNames();

        Thread t = new Thread(server);
        t.start();

        Queue<Integer> availablePorts = new LinkedList<>();

        for(int port : LocalTrainingServer.ports){
            availablePorts.add(port);
        }

        System.out.println("Accepting Clients");

        while(true) {
            int port = availablePorts.poll();
            ServerSocket ss = new ServerSocket(port);
            Socket socket = ss.accept();

            System.out.println("Client connected on port " + port);

            Runnable r = new Runnable() {
                @Override
                public void run() {
                    try {
                        OutputStream rawOutput = socket.getOutputStream();
                        ObjectOutputStream output = new ObjectOutputStream(rawOutput);
                        ObjectInputStream input = new ObjectInputStream(socket.getInputStream());

                        while (true) {
                            String message = (String) input.readObject();

                            switch (message) {
                                case ("addData"):
                                    try {
                                        INDArray[] startState = (INDArray[]) input.readObject();
                                        INDArray[] endState = (INDArray[]) input.readObject();
                                        INDArray[] masks = (INDArray[]) input.readObject();
                                        float score = (float) input.readObject();

                                        if(server.dataPoints.size() % 100 == 0){
                                            server.writeStateToImage(startState, "start");
                                        }

                                        server.addData(startState, endState, masks, score);
                                    }
                                    catch (Exception e){
                                        System.out.println("Error while attempting to upload a data point, point destroyed");
                                    }
                                    //while(server.pointsGathered > server.batchSize && server.iterations < server.pointsGathered){
                                    //    Thread.sleep(10);
                                    //}
                                    break;
                                case ("getUpdatedNetwork"):
                                    ComputationGraph graph = server.getUpdatedNetwork();
                                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                                    ModelSerializer.writeModel(graph, baos, false);

                                    output.writeObject(baos.toByteArray());
                                    break;
                                case ("getDependencyGraph"):
                                    output.writeObject(server.getDependencyGraph());
                                    break;
                                default:
                                    System.out.println("Got unregistered input " + message);
                            }
                        }
                    } catch (Exception e) {
                        System.out.println(e);
                        availablePorts.add(port);
                        System.out.println("Client disconnected from port " + port);
                    }
                }
            };

            Thread sockThread = new Thread(r);
            sockThread.start();
        }
    }

    @Override
    public void run() {
        while(this.run){
            System.out.print("");

            boolean sufficientDataGathered = this.dataPoints.size() > this.batchSize;

            if (sufficientDataGathered && pointWait > 0) {
                long start = System.currentTimeMillis();

                INDArray[][] startStates = new INDArray[this.batchSize][];
                INDArray[][] endStates = new INDArray[this.batchSize][];
                INDArray[][] labels = new INDArray[this.batchSize][];
                INDArray[][] masks = new INDArray[this.batchSize][];

                synchronized (this.dataPoints) {
                    for (int i = 0; i < this.batchSize; i++) {
                        int index = this.random.nextInt(this.dataPoints.size());
                        DataPoint data = this.dataPoints.get(index);

                        startStates[i] = data.getStartState();
                        endStates[i] = data.getEndState();
                        labels[i] = data.getLabels();
                        masks[i] = data.getMasks();
                    }
                }

                DataPoint cumulativeData = new DataPoint(this.concatSet(startStates), this.concatSet(endStates), this.concatSet(labels), this.concatSet(masks));

                long end = System.currentTimeMillis();
                System.out.print(end - start);

                start = System.currentTimeMillis();

                for(GraphMetadata metaData : this.graphs.keySet()) {
                    ComputationGraph graph = this.graphs.get(metaData);
                    ComputationGraph targetGraph = this.targetGraphs.get(metaData);

                    INDArray[] curLabels = graph.output(cumulativeData.getEndState());
                    INDArray[] targetLabels = targetGraph.output(cumulativeData.getEndState());

                    INDArray[] targetMaxs = new INDArray[curLabels.length];

                    for(ArrayList<Integer> inds : this.dependencyGraph.getAgentInds(this.outputs)){
                        int concatInd = 0;
                        INDArray[] curIndLabels = new INDArray[inds.size()];
                        for(int i : inds){
                            curIndLabels[concatInd] = curLabels[i];
                            concatInd++;
                            //max = max.add(curLabels[i]).add(abs(max.sub(curLabels[i]))).mul(.5);
                        }
                        INDArray max = Nd4j.concat(1, curIndLabels);
                        max = Nd4j.max(max, 1);

                        //BUG: .eq does not return 1 for all the correct values because the float values are changed too much in the max func
                        INDArray[] maxBools = new INDArray[curLabels.length];
                        for(int i : inds){
                            maxBools[i] = curLabels[i].eq(max);
                        }

                        INDArray targetMax = Nd4j.zeros(targetLabels[0].shape());
                        for(int i : inds){
                            targetMax = targetMax.add(maxBools[i].mul(targetLabels[i]));
                        }

                        for(int i : inds){
                            targetMaxs[i] = targetMax;
                        }
                    }

                    MultiDataSet dataSet = cumulativeData.getDataSetWithQOffset(targetMaxs, metaData.decayRate);
                    graph.fit(dataSet);

                    if (metaData.targetRotation != 0 && iterations % metaData.targetRotation == 0) {
                        this.targetGraphs.put(metaData, this.getUpdatedNetwork(metaData, true, false));
                    }

                    if(iterations % 100 == 0){
                        System.out.println(metaData.getName());
                    }
                }

                end = System.currentTimeMillis();
                System.out.println(" " + (end - start));

                iterations++;

            }
            else{
                try {
                    Thread.sleep(50);
                }
                catch (Exception e){
                    System.out.println("This thread is weak");
                }
            }

            this.pointWait -= 1;
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

        synchronized (this.dataPoints) {
            this.dataPoints.add(data);
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
        return this.getUpdatedNetwork(randomData, !this.connectFromNetwork, true);
    }

    private ComputationGraph getUpdatedNetwork(GraphMetadata metaData, boolean clone, boolean overwrite) {
        try{
            ComputationGraph graph = this.graphs.get(metaData);
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ModelSerializer.writeModel(graph, baos, true);

            byte[] modelBytes = baos.toByteArray();

            if(overwrite) {
                File f = new File(this.getModelName(metaData));

                if (f.exists()) {
                    f.delete();
                }

                FileOutputStream fout = new FileOutputStream(f);
                fout.write(modelBytes);
            }

            if(!clone){
                return graph;
            }
            else {
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
    public void stop() {
        this.run  = false;
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
        this.targetGraphs.put(metaData, this.getUpdatedNetwork(metaData, metaData.targetRotation != 0, false));

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new FileStatsStorage(new File(metaData.getName() + ".stor"));         //Alternative: new FileStatsStorage(File), for saving and loading later

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        //Then add the StatsListener to collect this information from the network, as it trains
        this.graphs.get(metaData).setListeners(/*new StatsListener(statsStorage),*/ new ScoreIterationListener(100));
    }

    private INDArray[] concatSet(INDArray[][] set){
        INDArray[] result = new INDArray[set[0].length];

        for(int j = 0; j < set[0].length; j++) {
            INDArray[] toConcat = new INDArray[set.length];

            for (int i = 0; i < set.length; i++) {
                toConcat[i] = set[i][j];
            }

            result[j] = Nd4j.concat(0, toConcat);
        }

        return result;
    }

    private void writeStateToImage(INDArray[] state, String fileName){
        try{
            File f = new File(fileName + ".jpg");
            BufferedImage img = new BufferedImage(this.inputSize, this.inputSize, BufferedImage.TYPE_3BYTE_BGR);

            INDArray stateImage = state[0].getColumn(3);
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
