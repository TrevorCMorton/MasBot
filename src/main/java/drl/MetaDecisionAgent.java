package drl;

import drl.agents.IAgent;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.*;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

public class MetaDecisionAgent {
    private ComputationGraph metaGraph;
    private AgentDependencyGraph dependencyGraph;
    private String[] outputs;
    private long iters;
    private double prob;
    private int commDepth;
    private ArrayList<ArrayList<Integer>> agentInds;

    public MetaDecisionAgent(AgentDependencyGraph dependencyGraph, double prob, boolean build, int commDepth){
        this.dependencyGraph = dependencyGraph;
        this.prob = prob;
        this.commDepth = commDepth;

        iters = 0;

        Nesterovs updater = new Nesterovs(.1);
        updater.setLearningRate(.0001);

        ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
            .seed(123)
            .l2(0.0005)
            .weightInit(WeightInit.XAVIER)
            .updater(updater)
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

        if(build) {
            ComputationGraphConfiguration conf = builder.build();
            metaGraph = new ComputationGraph(conf);
            metaGraph.init();
        }
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

            if(Math.random() > prob){
                actions[i] = agentOutputs.get((int)(Math.random() * agentOutputs.size()));
            }
            else {
                float best = 0.0f;
                String bestAction = "";
                for (String action : agentOutputs) {
                    System.out.print(action + ": " + outputValues.get(action) + " ");
                    if (bestAction.equals("") || outputValues.get(action) > best) {
                        bestAction = action;
                        best = outputValues.get(action);
                    }
                }
                System.out.println("Best: " + bestAction);
                actions[i] = bestAction;
            }
            i++;
        }

        if(iters % 100 == 0) {
            System.out.println(iters);
        }
        iters++;

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

    public String[] getOutputNames(){
        return this.outputs;
    }

    public ArrayList<ArrayList<Integer>> getAgentInds() { return this.agentInds; }

    private List<String> buildInputs(ComputationGraphConfiguration.GraphBuilder builder, int numActions){
        List<String> inputNames = new ArrayList<>();
        inputNames.add("Screen3");
        //inputNames.add("prevActions");

        builder.addInputs("Screen").setInputTypes(InputType.convolutionalFlat(84,84,4)/*, InputType.feedForward(numActions)*/);

        builder
                .addLayer("Screen1",
                new ConvolutionLayer.Builder(8, 8).nIn(4).stride(4, 4).nOut(32).activation(Activation.RELU).build(),
                "Screen")
                .addLayer("Screen2",
                        new ConvolutionLayer.Builder(4, 4).stride(2, 2).nOut(64).activation(Activation.RELU).build(),
                        "Screen1")
                .addLayer("Screen3",
                        new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(64).activation(Activation.RELU).build(),
                        "Screen2");

        return inputNames;
    }

    private List<String> buildHelper(AgentDependencyGraph.Node node, ComputationGraphConfiguration.GraphBuilder builder, List<String> envInputNames){
        if(node.built){
            return node.agent.getInternalOutputNames();
        }

        List<String> communicationOutputs = new ArrayList<>();

        for(AgentDependencyGraph.Node dependency : node.dependencies){
            List<String> dependencyOutputs = this.buildHelper(dependency, builder, envInputNames);
            communicationOutputs.addAll(buildCommunicationUnit(dependencyOutputs, node.agent.getName(), builder));
        }

        IAgent nodeAgent = node.agent;
        List<String> outputs = nodeAgent.build(builder, envInputNames, communicationOutputs);
        node.built = true;
        return outputs;
    }

    private List<String> buildCommunicationUnit(List<String> inputs, String name, ComputationGraphConfiguration.GraphBuilder builder){
        List<String> commOutputs = new ArrayList<>();

        String[] layerInputs = new String[inputs.size()];
        inputs.toArray(layerInputs);

        for(int i = 0; i < this.commDepth; i++) {
            String layerName = name + "Comm" + i;
            int nodeCount = -((inputs.size() - 1) / (this.commDepth - 1)) * i + inputs.size();
            commOutputs.add(layerName);
            builder.addLayer(layerName,
                    new DenseLayer.Builder().nOut(nodeCount).activation(Activation.IDENTITY).build(),
                    layerInputs);
            layerInputs = new String[] { layerName };
        }

        return commOutputs;
    }
}
