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
import org.nd4j.linalg.learning.config.Nesterovs;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class MetaDecisionAgent {

    private ComputationGraph metaGraph;
    private AgentDependencyGraph dependencyGraph;
    private String[] outputs;

    public MetaDecisionAgent(AgentDependencyGraph dependencyGraph){
        this.dependencyGraph = dependencyGraph;
        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
            .seed(123)
            .l2(0.0005)
            .weightInit(WeightInit.XAVIER)
            .updater(new Nesterovs.Builder().learningRate(.01).build())
            .biasUpdater(new Nesterovs.Builder().learningRate(0.02).build())
            .graphBuilder();

        // for now there is only one input, but we will eventually want to feed the last decisions mask vector as input
        List<String> inputs = this.buildInputs(builder);

        List<AgentDependencyGraph.Node> nodes = this.dependencyGraph.getSources();
        int index = 0;
        while(index != nodes.size()){
            AgentDependencyGraph.Node node = nodes.get(index);
            this.buildHelper(node, builder, inputs);
            nodes.addAll(node.dependents);
            index++;
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

        //Then add the StatsListener to collect this information from the network, as it trains
        metaGraph.setListeners(new StatsListener(statsStorage));
    }

    public String[] eval(INDArray[] features){
        List<AgentDependencyGraph.Node> nodes = this.dependencyGraph.getSources();
        int index = 0;
        while(index != nodes.size()){
            AgentDependencyGraph.Node node = nodes.get(index);
            nodes.addAll(node.dependents);
            index++;
        }

        INDArray[] results = metaGraph.output(features);

        HashMap<String, Float> outputValues = new HashMap<>();

        for(int i = 0; i < this.outputs.length; i++){
            outputValues.put(this.outputs[i], results[i].getFloat(0));
        }

        String[] actions = new String[nodes.size()];

        for(int i = 0; i < nodes.size(); i++){
            AgentDependencyGraph.Node node = nodes.get(i);
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
        }

        return actions;
    }

    private List<String> buildInputs(ComputationGraphConfiguration.GraphBuilder builder){
        List<String> inputNames = new ArrayList<>();
        inputNames.add("screen");

        builder.addInputs(inputNames).setInputTypes(InputType.convolutionalFlat(84,84,4));

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
