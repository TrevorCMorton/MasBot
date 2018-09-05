package agents;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

public class MeleeJoystickAgent implements IAgent{
    private List<String> outputNames;
    private String name;

    public MeleeJoystickAgent(String name){
        this.name = name;

        this.outputNames = new ArrayList<>();

        String[] outputNameStubs = { "R", "N", "NE", "E", "SE", "S", "SW", "W", "NW" };
        for(String outputName : outputNameStubs){
            this.outputNames.add(name + outputName);
        }
    }

    @Override
    public List<String> build(ComputationGraphConfiguration.GraphBuilder builder, List<String> envInputNames, List<String> dependencyInputNames) {
        builder
            .addLayer(this.name + "Joystick1",
                new ConvolutionLayer.Builder(8, 8).nIn(4).stride(4, 4).nOut(32).activation(Activation.RELU).build(),
                envInputNames.get(0))
            .addLayer(this.name + "Joystick2",
                new ConvolutionLayer.Builder(4, 4).stride(2, 2).nOut(64).activation(Activation.RELU).build(),
                    this.name + "Joystick1")
            .addLayer(this.name + "Joystick3",
                new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(64).activation(Activation.RELU).build(),
                    this.name + "Joystick2");

        builder
            .addLayer(this.name + "Joystick4",
                new DenseLayer.Builder().nOut(512).activation(Activation.RELU).build(),
                    this.name + "Joystick3");

        String[] mergeInputs = new String[dependencyInputNames.size() + envInputNames.size()];

        for(int i = 0; i < dependencyInputNames.size(); i++){
            mergeInputs[i] = dependencyInputNames.get(i);
        }

        for(int i = 1; i < envInputNames.size(); i++){
            mergeInputs[i + dependencyInputNames.size() - 1] = envInputNames.get(i);
        }
        mergeInputs[mergeInputs.length - 1] = this.name + "Joystick4";

        for(int i = 0; i < outputNames.size(); i++){
            String outputName = outputNames.get(i);

            builder
                .addLayer(outputName + "Internal",
                        new DenseLayer.Builder().nOut(1).activation(Activation.IDENTITY).build(),
                        mergeInputs)
                .addLayer(outputName,
                        new LossLayer.Builder().lossFunction(LossFunctions.LossFunction.SQUARED_LOSS).build(),
                        outputName + "Internal");
        }

        return this.getInternalOutputNames();
    }

    @Override
    public List<String> getOutputNames() {
        return this.outputNames;
    }

    @Override
    public List<String> getInternalOutputNames() {
        ArrayList<String> internalNames = new ArrayList<>();

        for(int i = 0; i < outputNames.size(); i++){
            String outputName = outputNames.get(i);
            internalNames.add(outputName + "Internal");
        }

        return internalNames;
    }
}
