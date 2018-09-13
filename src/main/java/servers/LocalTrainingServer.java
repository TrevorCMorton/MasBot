package servers;

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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Random;

public class LocalTrainingServer implements ITrainingServer{
    private ComputationGraph graph;
    private CircularFifoQueue<DataPoint> dataPoints;
    private int batchSize;
    private Random random;
    public int iterations;

    public LocalTrainingServer(ComputationGraph graph, int maxReplaySize, int batchSize){
        this.setNetwork(graph);
        dataPoints = new CircularFifoQueue<>(maxReplaySize);
        this.batchSize = batchSize;
        this.random = new Random(324);
        this.iterations = 0;
    }

    @Override
    public void run() {
        while(true){
            try {
                System.out.print("");
                if (this.dataPoints.size() > this.batchSize) {
                    INDArray[][] startStates = new INDArray[this.batchSize][];
                    INDArray[][] endStates = new INDArray[this.batchSize][];
                    INDArray[][] labels = new INDArray[this.batchSize][];
                    INDArray[][] masks = new INDArray[this.batchSize][];

                    synchronized (dataPoints) {
                        for (int i = 0; i < this.batchSize; i++) {
                            int index = this.random.nextInt(this.batchSize);
                            DataPoint data = this.dataPoints.get(index);

                            startStates[i] = data.getStartState();
                            endStates[i] = data.getEndState();
                            labels[i] = data.getLabels();
                            masks[i] = data.getMasks();
                        }
                    }

                    DataPoint cumulativeData = new DataPoint(this.concatSet(startStates), this.concatSet(endStates), this.concatSet(labels), this.concatSet(masks));

                    INDArray[] curLabels = graph.output(cumulativeData.getEndState());

                    MultiDataSet dataSet = cumulativeData.getDataSetWithQOffset(curLabels);

                    graph.fit(dataSet);
                    System.out.println("train");
                }
            }
            catch (Exception e){
                System.out.println(e);
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
    }

    @Override
    public ComputationGraph getUpdatedNetwork() {
        try{
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ModelSerializer.writeModel(this.graph, baos, true);

            ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
            return ModelSerializer.restoreComputationGraph(bais, true);
        }
        catch(Exception e){
            System.out.println(e);
        }
        return null;
    }

    @Override
    public void setNetwork(ComputationGraph graph) {
        try{
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ModelSerializer.writeModel(graph, baos, true);

            ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
            this.graph = ModelSerializer.restoreComputationGraph(bais, true);

            //Initialize the user interface backend
            UIServer uiServer = UIServer.getInstance();
            //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
            StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
            //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
            uiServer.attach(statsStorage);
            //Then add the StatsListener to collect this information from the network, as it trains
            this.graph.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(0));
        }
        catch(Exception e){
            System.out.println(e);
        }
    }

    @Override
    public int getDataSize(){
        return this.dataPoints.size();
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
