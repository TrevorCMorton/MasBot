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
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;
import java.util.*;

public class LocalTrainingServer implements ITrainingServer{
    public static final int[] ports = { 1612, 1613, 1614, 1615, 1616 };

    private ComputationGraph graph;
    private ComputationGraph targetGraph;
    private AgentDependencyGraph dependencyGraph;
    private MetaDecisionAgent agent;

    private boolean useWeightedTrainingPools;
    private CircularFifoQueue<DataPoint> neutralPoints;
    private CircularFifoQueue<DataPoint> posPoints;
    private CircularFifoQueue<DataPoint> negPoints;
    private CircularFifoQueue<DataPoint> posTermPoints;
    private CircularFifoQueue<DataPoint> negTermPoints;

    private CircularFifoQueue<DataPoint> dataPoints;
    private int batchSize;
    private Random random;
    private float decayRate;
    private int targetRotation;
    private boolean connectFromNetwork;
    private int pointsGathered;
    private int iterations;
    private boolean run;

    public LocalTrainingServer(boolean connectFromNetwork, AgentDependencyGraph dependencyGraph, int maxReplaySize, int batchSize, float decayRate, boolean useWeightedTrainingPools, int targetRotation){
        this.connectFromNetwork = connectFromNetwork;
        this.dataPoints = new CircularFifoQueue<>(maxReplaySize);
        this.batchSize = batchSize;
        this.random = new Random(324);
        this.decayRate = decayRate;
        this.targetRotation = targetRotation;

        this.dependencyGraph = dependencyGraph;
        this.agent = new MetaDecisionAgent(dependencyGraph, 0, true);
        this.graph = this.agent.getMetaGraph();
        this.targetGraph = this.getUpdatedNetwork(targetRotation != 0);
        this.graph.setListeners(new ScoreIterationListener(100));

        this.run = true;

        this.useWeightedTrainingPools = useWeightedTrainingPools;
        this.posPoints = new CircularFifoQueue<>(maxReplaySize);
        this.posTermPoints = new CircularFifoQueue<>(maxReplaySize);
        this.negPoints = new CircularFifoQueue<>(maxReplaySize);
        this.negTermPoints = new CircularFifoQueue<>(maxReplaySize);
        this.neutralPoints = new CircularFifoQueue<>(maxReplaySize);
    }

    public static void main(String[] args) throws Exception{
        CudaEnvironment.getInstance().getConfiguration()
                .allowMultiGPU(false)
                .allowCrossDeviceAccess(false)
                .setMaximumDeviceCache(8L * 1024L * 1024L * 1024L);

        AgentDependencyGraph dependencyGraph = new AgentDependencyGraph();

        IAgent joystickAgent = new MeleeJoystickAgent("M");
        IAgent cstickAgent = new MeleeJoystickAgent("C");
        IAgent abuttonAgent = new MeleeButtonAgent("A");
        dependencyGraph.addAgent(null, joystickAgent, "M");
        //dependencyGraph.addAgent(new String[]{"M"}, cstickAgent, "C");
        dependencyGraph.addAgent(new String[]{"M"}, abuttonAgent, "A");

        int replaySize = Integer.parseInt(args[0]);
        int batchSize = Integer.parseInt(args[1]);
        float decayRate = Float.parseFloat(args[2]);
        boolean useWeightedTrainingPools = Boolean.parseBoolean(args[3]);
        int targetRotation = Integer.parseInt(args[4]);

        LocalTrainingServer server = new LocalTrainingServer(true, dependencyGraph, replaySize, batchSize, decayRate, useWeightedTrainingPools, targetRotation);

        File pretrained = new File(server.getModelName());

        if(pretrained.exists()){
            System.out.println("Loading model from file");
            ComputationGraph model = ModelSerializer.restoreComputationGraph(pretrained, true);
            server.setGraph(model);
        }

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

                                        server.addData(startState, endState, masks, score);
                                    }
                                    catch (Exception e){
                                        System.out.println("Error while attempting to upload a data point, point destroyed");
                                    }
                                    while(server.pointsGathered > server.batchSize && server.iterations < server.pointsGathered){
                                        Thread.sleep(10);
                                    }
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
        ArrayList<CircularFifoQueue<DataPoint>> pools = new ArrayList<>();
        pools.add(this.neutralPoints);
        pools.add(this.negPoints);
        pools.add(this.negTermPoints);
        pools.add(this.posPoints);
        pools.add(this.posTermPoints);

        while(this.run){
            System.out.print("");
            pools.sort(new Comparator<CircularFifoQueue<DataPoint>>() {
                @Override
                public int compare(CircularFifoQueue<DataPoint> o1, CircularFifoQueue<DataPoint> o2) {
                    return o1.size() - o2.size();
                }
            });

            boolean sufficientDataGathered = this.dataPoints.size() > this.batchSize;

            if (sufficientDataGathered) {
                INDArray[][] startStates = new INDArray[this.batchSize][];
                INDArray[][] endStates = new INDArray[this.batchSize][];
                INDArray[][] labels = new INDArray[this.batchSize][];
                INDArray[][] masks = new INDArray[this.batchSize][];

                if(useWeightedTrainingPools){
                    int curInd = 0;

                    for(int j = 0; j < pools.size(); j++){
                        synchronized (pools.get(j)){
                            int endInd = this.batchSize / (pools.size() - j) - curInd > pools.get(j).size() ? curInd + pools.get(j).size() : curInd + this.batchSize / (pools.size() - j) - curInd;
                            for (int i = curInd; i < endInd; i++) {
                                int index = this.random.nextInt(pools.get(j).size());
                                DataPoint data = pools.get(j).get(index);

                                startStates[i] = data.getStartState();
                                endStates[i] = data.getEndState();
                                labels[i] = data.getLabels();
                                masks[i] = data.getMasks();

                                curInd++;
                            }
                        }
                    }

                    if(this.iterations % 100 == 0){
                        System.out.println(this.neutralPoints.size() + " " + this.negPoints.size() + " " + this.negTermPoints.size() + " " + this.posPoints.size() + " " + this.posTermPoints.size());
                    }
                }
                else {
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
                }

                DataPoint cumulativeData = new DataPoint(this.concatSet(startStates), this.concatSet(endStates), this.concatSet(labels), this.concatSet(masks));

                INDArray[] curLabels = targetGraph.output(cumulativeData.getEndState());

                MultiDataSet dataSet = cumulativeData.getDataSetWithQOffset(curLabels, decayRate);

                graph.fit(dataSet);

                iterations++;

                if(targetRotation != 0 && iterations % this.targetRotation == 0){
                    this.targetGraph = this.getUpdatedNetwork(true);
                }
            }
            else{
                try {
                    Thread.sleep(1000);
                }
                catch (Exception e){
                    System.out.println("This thread is weak");
                }
            }
        }
    }

    @Override
    public void addData(INDArray[] startState, INDArray[] endState, INDArray[] masks, float score) {
        INDArray[] labels = new INDArray[masks.length];

        for (int i = 0; i < masks.length; i++) {
            labels[i] = masks[i].mul(score);
        }

        DataPoint data = new DataPoint(startState, endState, labels, masks);

        if(this.useWeightedTrainingPools){
            if(score == 0){
                synchronized (this.neutralPoints) {
                    this.neutralPoints.add(data);
                }
            }
            else if(score > 0){
                if(score >= 1){
                    synchronized (this.posTermPoints) {
                        this.posTermPoints.add(data);
                    }
                }
                else{
                    synchronized (this.posPoints) {
                        this.posPoints.add(data);
                    }
                }
            }
            else{
                if(score <= -1){
                    synchronized (this.negTermPoints) {
                        this.negTermPoints.add(data);
                    }
                }
                else{
                    synchronized (this.negPoints) {
                        this.negPoints.add(data);
                    }
                }
            }
        }
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
        return this.getUpdatedNetwork(!this.connectFromNetwork);
    }

    private ComputationGraph getUpdatedNetwork(boolean clone) {
        try{
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ModelSerializer.writeModel(this.graph, baos, true);

            byte[] modelBytes = baos.toByteArray();

            File f = new File(this.getModelName());

            if(f.exists()){
                f.delete();
            }

            FileOutputStream fout = new FileOutputStream(f);
            fout.write(modelBytes);

            if(!clone){
                return this.graph;
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

    protected String getModelName(){
        StringBuilder sb = new StringBuilder();
        sb.append("model-");
        for(String output : this.agent.getOutputNames()){
            sb.append(output);
            sb.append("-");
        }
        sb.append(dataPoints.maxSize());
        sb.append("-");
        sb.append(this.batchSize);
        sb.append("-");
        sb.append(this.decayRate);
        sb.append("-");
        sb.append(MetaDecisionAgent.commDepth);
        sb.append(".mod");
        return sb.toString();
    }

    protected void setGraph(ComputationGraph graph){
        this.graph = graph;
        this.targetGraph = this.getUpdatedNetwork(this.targetRotation != 0);
        this.graph.setListeners(new ScoreIterationListener(100));
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
}
