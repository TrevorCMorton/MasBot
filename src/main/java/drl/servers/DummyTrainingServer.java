package drl.servers;

import drl.AgentDependencyGraph;
import drl.MetaDecisionAgent;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

public class DummyTrainingServer implements ITrainingServer {
    private ComputationGraph graph;
    private AgentDependencyGraph agentDependencyGraph;

    public DummyTrainingServer(AgentDependencyGraph dependencyGraph) {
        this.agentDependencyGraph = dependencyGraph;
        MetaDecisionAgent agent = new MetaDecisionAgent(dependencyGraph, .5, true, 3);
        this.graph = agent.getMetaGraph();
    }

    @Override
    public void addData(INDArray[] startState, INDArray[] endState, INDArray[] masks, float score) {

    }

    @Override
    public ComputationGraph getUpdatedNetwork() {
        return this.graph;
    }

    @Override
    public AgentDependencyGraph getDependencyGraph() {
        return this.agentDependencyGraph;
    }

    @Override
    public void stop() {

    }

    @Override
    public void run() {

    }
}
