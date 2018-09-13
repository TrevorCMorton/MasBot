import agents.IAgent;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

public class MetaDecisionAgent {

    private ComputationGraph metaGraph;
    private AgentDependencyGraph dependencyGraph;
    private String[] outputs;

    public MetaDecisionAgent(AgentDependencyGraph dependencyGraph){
        this.dependencyGraph = dependencyGraph;

        ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
            .seed(123)
            .l2(0.0005)
            .weightInit(WeightInit.XAVIER)
            .updater(new Nesterovs.Builder().learningRate(.01).build())
            .biasUpdater(new Nesterovs.Builder().learningRate(0.02).build())
            .graphBuilder();

        Collection<AgentDependencyGraph.Node> nodes = this.dependencyGraph.getNodes();

        int numActions = 0;
        for(AgentDependencyGraph.Node node : nodes){
            numActions += node.agent.getOutputNames().size();
        }

        List<String> inputs = this.buildInputs(builder, numActions);

        for(AgentDependencyGraph.Node node : nodes){
            this.buildHelper(node, builder, inputs);
        }

        List<String> outputNames = new ArrayList<>();

        for(AgentDependencyGraph.Node node : nodes){
            outputNames.addAll(node.agent.getOutputNames());
        }

        this.outputs = new String[outputNames.size()];

        for(int i = 0; i < outputNames.size(); i++){
            this.outputs[i] = outputNames.get(i);
        }

        builder.setOutputs(this.outputs);

        ComputationGraphConfiguration conf = builder.build();
        metaGraph = new ComputationGraph(conf);
        metaGraph.init();
    }

    public String[] eval(INDArray[] features){
        Collection<AgentDependencyGraph.Node> nodes = this.dependencyGraph.getNodes();

        INDArray[] results = metaGraph.output(features);

        HashMap<String, Float> outputValues = new HashMap<>();

        for(int i = 0; i < this.outputs.length; i++){
            outputValues.put(this.outputs[i], results[i].getFloat(0));
        }

        String[] actions = new String[nodes.size()];
        int i = 0;

        for(AgentDependencyGraph.Node node : nodes){
            List<String> agentOutputs = node.agent.getOutputNames();

            float best = 0.0f;
            String bestAction = "";
            for(String action : agentOutputs){
                if(bestAction.equals("") || outputValues.get(action) > best){
                    bestAction = action;
                    best = outputValues.get(action);
                }
            }

            actions[i] = bestAction;
            i++;
        }

        return actions;
    }

    public INDArray[] getOutputMask(String[] actions){
        INDArray[] masks = new INDArray[this.outputs.length];
        for(int i = 0; i < this.outputs.length; i++){
            INDArray maskEntry = Nd4j.zeros(1);
            for(int j = 0; j < actions.length; j++){
                if(this.outputs[i].equals(actions[j])){
                    maskEntry = Nd4j.ones(1);
                }
            }
            masks[i] = maskEntry;
        }
        return masks;
    }

    public ComputationGraph getMetaGraph() {
        return this.metaGraph;
    }

    public void setMetaGraph(ComputationGraph graph){
        this.metaGraph = graph;
    }

    public int getNumOutputs(){
        return metaGraph.getNumOutputArrays();
    }

    private List<String> buildInputs(ComputationGraphConfiguration.GraphBuilder builder, int numActions){
        List<String> inputNames = new ArrayList<>();
        inputNames.add("screen");
        //inputNames.add("prevActions");

        builder.addInputs(inputNames).setInputTypes(InputType.convolutionalFlat(84,84,4)/*, InputType.feedForward(numActions)*/);

        return inputNames;
    }

    private List<String> buildHelper(AgentDependencyGraph.Node node, ComputationGraphConfiguration.GraphBuilder builder, List<String> envInputNames){
        if(node.built){
            return node.agent.getInternalOutputNames();
        }

        List<String> dependencyOutputs = new ArrayList<>();

        for(AgentDependencyGraph.Node dependency : node.dependencies){
            dependencyOutputs.addAll(this.buildHelper(dependency, builder, envInputNames));
        }

        IAgent nodeAgent = node.agent;
        List<String> outputs = nodeAgent.build(builder, envInputNames, dependencyOutputs);
        node.built = true;
        return outputs;
    }
}
