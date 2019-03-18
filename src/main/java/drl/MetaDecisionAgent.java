package drl;

import drl.agents.IAgent;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.*;

import java.util.*;

public class MetaDecisionAgent {
    public static final int size = 168;
    public static final int depth = 1;

    private ComputationGraph metaGraph;
    private AgentDependencyGraph dependencyGraph;
    private String[] outputs;
    private ArrayList<String> inputs;
    private HashMap<Integer, String> orphanInputs;
    private ArrayList<InputType> types;
    private long iters;
    private double prob;
    private INDArray[] cachedLabels;
    private boolean makeOutputs;

    private long evals;
    private long initSetupTime;
    private long outputTime;
    private long layerSetupTime;
    private long layerDecisionTime;
    private long layerCleanupTime;

    private List<IActivation> activations;

    public MetaDecisionAgent(AgentDependencyGraph dependencyGraph, List<IActivation> activations, double prob, double learningRate, boolean makeOutputs){
        this.dependencyGraph = dependencyGraph;
        this.activations = activations;
        this.prob = prob;
        this.makeOutputs = makeOutputs;

        iters = 0;

        this.inputs = new ArrayList<>();
        this.orphanInputs = new HashMap<>();
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
            .updater(new RmsProp(learningRate))
            //.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1)
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

        for(int i = 0; i < this.outputs.length; i++){
            boolean contains = false;
            for(int j = 0; j < this.inputs.size(); j++){
                if(this.outputs[i].equals(this.inputs.get(j).substring(0, this.inputs.get(j).length() - 2))){
                    contains = true;
                    break;
                }
            }

            if(!contains){
                this.orphanInputs.put(i, this.outputs[i]);
            }
        }

        builder.setOutputs(this.outputs);

        InputType[] inputTypes = new InputType[this.types.size()];
        types.toArray(inputTypes);
        builder.setInputTypes(inputTypes);

        ComputationGraphConfiguration conf = builder.build();
        metaGraph = new ComputationGraph(conf);
        metaGraph.init();
    }

    public String[] evalState(INDArray[] state){
        INDArray[] outputs = this.getMetaGraph().output(state);

        ArrayList<String> actions = new ArrayList<>();

        for(int j = 1; j < this.inputs.size(); j++){
            String inputName = this.inputs.get(j);
            if(state[j].getFloat(0) == 1){
                actions.add(inputName.substring(0, inputName.length() - 2));
            }
        }

        ArrayList<ArrayList<Integer>> agentInds = this.dependencyGraph.getAgentInds(this.outputs);

        for(ArrayList<Integer> agentInd : agentInds){
            float max = outputs[agentInd.get(0)].getFloat(0);
            int best = agentInd.get(0);
            for(int index : agentInd){
                float val = outputs[index].getFloat(0);
                System.out.print(this.outputs[index] + ": " + val + " ");
                if(val > max){
                    max = val;
                    best = index;
                }
            }

            if(Math.random() > prob){
                best = agentInd.get((int)(Math.random() * agentInd.size()));
            }

            if(this.orphanInputs.containsKey(agentInd.get(0))) {
                actions.add(this.outputs[best]);
            }

            System.out.println(" Best: " + this.outputs[best]);
        }

        String[] actionArray = new String[actions.size()];
        actions.toArray(actionArray);

        return actionArray;
    }

    public INDArray[] getState(INDArray input){
        // dependency graph guarantees this will be in safe order
        ArrayList<ArrayList<Integer>> agentInds = this.dependencyGraph.getAgentInds(this.outputs);

        INDArray[] state = new INDArray[this.inputs.size()];
        state[0] = input;

        for(int i = 1; i < state.length; i++){
            state[i] = Nd4j.zeros(input.shape()[0], 1);
        }

        for(int i = 0; i < agentInds.size(); i++){
            INDArray[] outputs = this.getMetaGraph().output(state);

            ArrayList<Integer> agentInd = agentInds.get(i);
            INDArray[] agentOutputs = new INDArray[agentInd.size()];

            for(int j = 0; j < agentInd.size(); j++){
                agentOutputs[j] = outputs[agentInd.get(j)];
            }

            INDArray concatArray = Nd4j.concat(1, agentOutputs);
            INDArray max = concatArray.max(1);


            if(Math.random() > this.prob){
                max = concatArray.getColumn((int)(Math.random() * agentInd.size()));
            }

            INDArray[] masksConcat = new INDArray[agentOutputs.length];
            for(int j = 0; j < masksConcat.length; j++){
                masksConcat[j] = agentOutputs[j].eq(max);
                max = max.add(masksConcat[j]);
            }

            INDArray masks = Nd4j.concat(1, masksConcat);

            for(int j = 0; j < this.inputs.size(); j++){
                String inputName = this.inputs.get(j);
                for(int k = 0; k < agentInd.size(); k++){
                    if(inputName.substring(0, inputName.length() - 2).equals(this.outputs[agentInd.get(k)])){
                        state[j] = masks.getColumn(k).reshape(input.shape()[0], 1);
                        break;
                    }
                }
            }

            this.cachedLabels = outputs;
        }

        return state;
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

    public INDArray[] getCachedLabels() {
        return cachedLabels;
    }

    public void printEvalSummary(){
        System.out.println("Average time for eval initialization: " + (initSetupTime / evals) + "ms");
        System.out.println("Average time for model output: " + (outputTime / evals) + "ms");
        System.out.println("Average time for layer setup: " + (layerSetupTime / evals) + "ms");
        System.out.println("Average time for layer decisions: " + (layerDecisionTime / evals) + "ms");
        System.out.println("Average time for layer cleanup: " + (layerCleanupTime / evals) + "ms");
    }

    private List<String> buildEnvironmentInputs(ComputationGraphConfiguration.GraphBuilder builder, int numActions){
        this.addInput(builder, "Screen", InputType.convolutionalFlat(MetaDecisionAgent.size, MetaDecisionAgent.size, MetaDecisionAgent.depth));

        IActivation activation1;
        IActivation activation2;
        IActivation activation3;

        if(!this.makeOutputs){
            activation1 = new WeightedActivationRelu();
            activation2 = new WeightedActivationRelu();
            activation3 = new WeightedActivationRelu();
        }
        else{
            activation1 = new ActivationReLU();
            activation2 = new ActivationReLU();
            activation3 = new ActivationReLU();
        }
        this.activations.add(activation1);
        this.activations.add(activation2);
        this.activations.add(activation3);

        int convOutSize = ((((MetaDecisionAgent.size - 8) / 4 + 1) - 4) / 2 + 1);// - 2;

        builder
                /*.addLayer("Screen1",
                        new ConvolutionLayer.Builder(8, 8).nIn(MetaDecisionAgent.depth).stride(4, 4).nOut(32).weightInit(WeightInit.XAVIER).activation(activation1).build(),
                        "Screen")
                .addLayer("Screen2",
                        new ConvolutionLayer.Builder(4, 4).stride(2, 2).nOut(64).weightInit(WeightInit.XAVIER).activation(activation2).build(),
                        "Screen1")
                .addLayer("Screen3",
                        new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(64).weightInit(WeightInit.XAVIER).activation(activation3).build(),
                        "Screen2")
                .addVertex("Screen3Flat",
                    new PreprocessorVertex(new CnnToFeedForwardPreProcessor(convOutSize, convOutSize, 64)),
                    "Screen3");*/
                .addLayer("Screen1",
                        new ConvolutionLayer.Builder(8, 8).nIn(MetaDecisionAgent.depth).stride(4, 4).nOut(32).weightInit(WeightInit.XAVIER).activation(activation1).build(),
                        "Screen")
                .addLayer("Screen2",
                        new ConvolutionLayer.Builder(4, 4).stride(2, 2).nOut(64).weightInit(WeightInit.XAVIER).activation(activation2).build(),
                        "Screen1")
                .addVertex("Screen2Flat",
                        new PreprocessorVertex(new CnnToFeedForwardPreProcessor(convOutSize, convOutSize, 64)),
                        "Screen2");

        List<String> inputNames = new ArrayList<>();
        inputNames.add("Screen2Flat");

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
        List<String> outputs = nodeAgent.build(builder, envInputNames, dependencyNames, this.makeOutputs, this.activations);

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
