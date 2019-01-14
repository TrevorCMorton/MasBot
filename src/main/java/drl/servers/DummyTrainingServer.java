package drl.servers;

import drl.AgentDependencyGraph;
import drl.MetaDecisionAgent;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;

public class DummyTrainingServer implements ITrainingServer {
    private ComputationGraph graph;
    private AgentDependencyGraph agentDependencyGraph;

    public DummyTrainingServer(AgentDependencyGraph dependencyGraph, String filePath) throws Exception{
        this.agentDependencyGraph = dependencyGraph;

        File pretrained = new File(filePath);

        if(pretrained.exists()){
            System.out.println("Loading model from file");
            ComputationGraph model = ModelSerializer.restoreComputationGraph(pretrained, true);
            this.graph = model;
            System.out.println(model.summary());
        }
        else{
            MetaDecisionAgent agent = new MetaDecisionAgent(dependencyGraph, this.getProb());
            this.graph = agent.getMetaGraph();
            dependencyGraph.resetNodes();
            System.out.println(agent.getMetaGraph().summary());
        }
    }

    @Override
    public void addData(INDArray[] startState, INDArray[] endState, INDArray[] masks, float score, INDArray[] startLabels, INDArray[] endLabels) {

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
    public void pause() {

    }

    @Override
    public void resume() {

    }

    @Override
    public void stop() {

    }

    @Override
    public double getProb() {
        return 1;
    }

    @Override
    public void run() {

    }
}
