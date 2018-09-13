package servers;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface ITrainingServer extends Runnable{
    void addData(INDArray[] startState, INDArray[] endState, INDArray[] masks, float score);
    ComputationGraph getUpdatedNetwork();
    void setNetwork(ComputationGraph graph);
    int getDataSize();
}
