package drl.servers;

import drl.AgentDependencyGraph;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.rmi.Remote;
import java.rmi.RemoteException;

public interface ITrainingServer extends Runnable {
    void addData(INDArray[] startState, INDArray[] endState, INDArray[] masks, float score, INDArray[] startLabels, INDArray[] endLabels);
    ComputationGraph getUpdatedNetwork();
    AgentDependencyGraph getDependencyGraph();
    void pause();
    void resume();
    void stop();
    double getProb();
}
