package drl.servers;

import drl.AgentDependencyGraph;
import drl.MetaDecisionAgent;
import drl.agents.IAgent;
import drl.agents.MeleeJoystickAgent;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
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
import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;

public class LocalTrainingServer implements ITrainingServer{
    public static final int[] ports = { 1612, 1613, 1614, 1615, 1616 };

    private ComputationGraph graph;
    private AgentDependencyGraph dependencyGraph;

    private CircularFifoQueue<DataPoint> dataPoints;
    private int batchSize;
    private Random random;
    private float decayRate;
    private boolean connectFromNetwork;
    private int pointsGathered;
    private int iterations;
    private boolean run;

    public LocalTrainingServer(boolean connectFromNetwork, AgentDependencyGraph dependencyGraph, int maxReplaySize, int batchSize, float decayRate){
        this.connectFromNetwork = connectFromNetwork;

        this.dependencyGraph = dependencyGraph;
        MetaDecisionAgent decisionAgent = new MetaDecisionAgent(dependencyGraph, true);
        this.graph = decisionAgent.getMetaGraph();
        /*
        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();
        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);
        //Then add the StatsListener to collect this information from the network, as it trains
        this.graph.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(100));
        */
        this.graph.setListeners(new ScoreIterationListener(100));
        this.dataPoints = new CircularFifoQueue<>(maxReplaySize);
        this.batchSize = batchSize;
        this.random = new Random(324);
        this.decayRate = decayRate;
        this.run = true;
    }

    public static void main(String[] args) throws Exception{
        AgentDependencyGraph dependencyGraph = new AgentDependencyGraph();

        IAgent joystickAgent = new MeleeJoystickAgent("M");
        IAgent cstickAgent = new MeleeJoystickAgent("C");
        dependencyGraph.addAgent(null, joystickAgent, "M");
        //dependencyGraph.addAgent(new String[]{"M"}, cstickAgent, "C");

        int replaySize = Integer.parseInt(args[0]);
        int batchSize = Integer.parseInt(args[1]);
        float decayRate = Float.parseFloat(args[2]);

        ITrainingServer server = new LocalTrainingServer(true, dependencyGraph, replaySize, batchSize, decayRate);

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
            if (this.dataPoints.size() > this.batchSize && iterations <= pointsGathered) {
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

                INDArray[] curLabels = graph.output(cumulativeData.getEndState());

                MultiDataSet dataSet = cumulativeData.getDataSetWithQOffset(curLabels, decayRate);

                graph.getInputs();

                graph.fit(dataSet);

                iterations++;
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

        synchronized (this.dataPoints) {
            dataPoints.add(data);
        }

        pointsGathered++;

        if(pointsGathered % 100 == 0){
            System.out.println(pointsGathered + " points gathered");
        }
    }

    @Override
    public ComputationGraph getUpdatedNetwork() {
        try{
            if(connectFromNetwork){
                return this.graph;
            }
            else {
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                ModelSerializer.writeModel(this.graph, baos, true);

                ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
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
