package drl;

import drl.agents.IAgent;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
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
    public static final int size = 84;
    public static final int depth = 4;

    private ComputationGraph metaGraph;
    private AgentDependencyGraph dependencyGraph;
    private String[] outputs;
    private ArrayList<String> inputs;
    private ArrayList<InputType> types;
    private long iters;
    private double prob;
    private ArrayList<ArrayList<Integer>> agentInds;

    private long evals;
    private long initSetupTime;
    private long outputTime;
    private long layerSetupTime;
    private long layerDecisionTime;
    private long layerCleanupTime;

    public MetaDecisionAgent(AgentDependencyGraph dependencyGraph, double prob){
        this.dependencyGraph = dependencyGraph;
        this.prob = prob;

        iters = 0;

        this.inputs = new ArrayList<>();
        this.types = new ArrayList<>();

        this.dependencyGraph.resetNodes();

        evals = 0;
        initSetupTime = 0;
        outputTime = 0;
        layerSetupTime = 0;
        layerDecisionTime = 0;
        layerCleanupTime = 0;

        ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
            .seed(123)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Adam(.0001))
            .graphBuilder();

        Collection<AgentDependencyGraph.Node> nodes = this.dependencyGraph.getNodes();

        int numActions = 0;
        for(AgentDependencyGraph.Node node : nodes){
            numActions += node.agent.getOutputNames().size();
        }

        List<String> inputs = this.buildEnvironmentInputs(builder, numActions);

        for(AgentDependencyGraph.Node node : nodes){
            this.buildHelper(node, builder, inputs);
            //if(node.dependents.size() == 0){
            //    for(String name : node.agent.getOutputNames()){
            //        builder.removeVertex(name + "In");
            //    }
            //}
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

        InputType[] inputTypes = new InputType[this.types.size()];
        types.toArray(inputTypes);
        builder.setInputTypes(inputTypes);

        ComputationGraphConfiguration conf = builder.build();
        metaGraph = new ComputationGraph(conf);
        metaGraph.init();
    }

    public INDArray[] getState(INDArray frame, String[] results){
        INDArray[] graphInputs = new INDArray[this.inputs.size()];

        graphInputs[0] = frame;

        for(int i = 1; i < this.inputs.size(); i++){
            for(int j = 0; j < results.length; j++){
                if(this.inputs.get(i).startsWith(results[j])){
                    graphInputs[i] = Nd4j.ones(1);
                }
            }

            if(graphInputs[i] == null){
                graphInputs[i] = Nd4j.zeros(1);
            }
        }

        return graphInputs;
    }

    public String[] eval(INDArray input){
        this.evals++;

        long start = System.currentTimeMillis();

        ArrayList<String> actions = new ArrayList<>();
        INDArray[] features = this.getState(input, new String[] {} );
        INDArray[] results;

        HashMap<String, Float> outputValues = new HashMap<>();
        List<AgentDependencyGraph.Node> nodeLayer = this.dependencyGraph.getRoots();

        long setup = System.currentTimeMillis();
        this.initSetupTime += setup - start;

        while(!nodeLayer.isEmpty()) {
            long layerStart = System.currentTimeMillis();

            results = metaGraph.output(features);

            long layerOutputTime = System.currentTimeMillis();
            this.outputTime += layerOutputTime - layerStart;

            // Populate map of output names to value
            for(int i = 0; i < this.outputs.length; i++){
                outputValues.put(this.outputs[i], results[i].getFloat(0));
            }

            long layerSetup = System.currentTimeMillis();
            this.layerSetupTime += layerSetup - layerOutputTime;

            // Choose action from each node, either random or the highest value
            for(AgentDependencyGraph.Node node : nodeLayer) {
                List<String> nodeOutputs = node.agent.getOutputNames();
                String chosenAction;
                if (Math.random() > prob) {
                    chosenAction = nodeOutputs.get((int) (Math.random() * nodeOutputs.size()));
                } else {
                    float best = outputValues.get(nodeOutputs.get(0));
                    String bestAction = nodeOutputs.get(0);
                    for (String action : nodeOutputs) {
                        System.out.print(action + ": " + outputValues.get(action) + " ");
                        if (outputValues.get(action) > best) {
                            bestAction = action;
                            best = outputValues.get(action);
                        }
                    }
                    System.out.println("Best: " + bestAction);
                    chosenAction = bestAction;
                }

                actions.add(chosenAction);
            }


            long layerDecision = System.currentTimeMillis();
            this.layerDecisionTime += layerDecision - layerSetup;

            // For each node add dependents to a new list
            List<AgentDependencyGraph.Node> nextLayer = new ArrayList<>();
            for (AgentDependencyGraph.Node node : nodeLayer) {
                nextLayer.addAll(node.dependents);
            }

            // Setup next layer
            nodeLayer = nextLayer;

            // Convert chosen actions to array, get state based on chosen actions and reevaluate network
            String[] actionArray = new String[actions.size()];
            actions.toArray(actionArray);
            features = this.getState(input, actionArray);

            long layerCleanup = System.currentTimeMillis();
            layerCleanupTime += layerCleanup - layerDecision;
        }

        if(iters % 100 == 0) {
            System.out.println(iters);
        }
        iters++;

        String[] actionArray = new String[actions.size()];
        actions.toArray(actionArray);
        return actionArray;
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

    public void printEvalSummary(){
        System.out.println("Average time for eval initialization: " + (initSetupTime / evals) + "ms");
        System.out.println("Average time for model output: " + (outputTime / evals) + "ms");
        System.out.println("Average time for layer setup: " + (layerSetupTime / evals) + "ms");
        System.out.println("Average time for layer decisions: " + (layerDecisionTime / evals) + "ms");
        System.out.println("Average time for layer cleanup: " + (layerCleanupTime / evals) + "ms");
    }

    private List<String> buildEnvironmentInputs(ComputationGraphConfiguration.GraphBuilder builder, int numActions){
        this.addInput(builder, "Screen", InputType.convolutionalFlat(MetaDecisionAgent.size, MetaDecisionAgent.size,4));

        int convOutSize = ((((MetaDecisionAgent.size - 8) / 4 + 1) - 4) / 2 + 1) - 2;

        builder
                .addLayer("Screen1",
                        new ConvolutionLayer.Builder(8, 8).nIn(4).stride(4, 4).nOut(32).weightInit(WeightInit.XAVIER).activation(Activation.RELU).build(),
                        "Screen")
                .addLayer("Screen2",
                        new ConvolutionLayer.Builder(4, 4).stride(2, 2).nOut(64).weightInit(WeightInit.XAVIER).activation(Activation.RELU).build(),
                        "Screen1")
                .addLayer("Screen3",
                        new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(64).weightInit(WeightInit.XAVIER).activation(Activation.RELU).build(),
                        "Screen2")
                .addVertex("Screen3Flat",
                    new PreprocessorVertex(new CnnToFeedForwardPreProcessor(convOutSize, convOutSize, 64)),
                    "Screen3");

        List<String> inputNames = new ArrayList<>();
        inputNames.add("Screen3Flat");

        return inputNames;
    }

    private List<String> buildHelper(AgentDependencyGraph.Node node, ComputationGraphConfiguration.GraphBuilder builder, List<String> envInputNames){
        if(node.built){
            return node.agent.getOutputNames();
        }

        List<String> dependencyNames = new ArrayList<>();

        for(AgentDependencyGraph.Node dependency : node.dependencies){
            List<String> dependencyOutputs = this.buildHelper(dependency, builder, envInputNames);
            dependencyNames.addAll(dependencyOutputs);
        }

        IAgent nodeAgent = node.agent;
        List<String> outputs = nodeAgent.build(builder, envInputNames, dependencyNames);

        if(node.dependents.size() != 0) {
            for (String output : outputs) {
                this.addInput(builder, output + "In", InputType.feedForward(1));
            }
        }

        node.built = true;
        return outputs;
    }

    private void addInput(ComputationGraphConfiguration.GraphBuilder builder, String inputName, InputType type){
        this.inputs.add(inputName);
        this.types.add(type);
        builder.addInputs(inputName);
    }
}
