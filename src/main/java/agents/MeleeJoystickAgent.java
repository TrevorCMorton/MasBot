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
    private List<String> internalOutputNames;

    @Override
    public List<String> build(ComputationGraphConfiguration.GraphBuilder builder, List<String> envInputNames, List<String> dependencyInputNames) {
        builder
            .addLayer("Joystick1",
                new ConvolutionLayer.Builder(8, 8).nIn(4).stride(4, 4).nOut(32).activation(Activation.RELU).build(),
                envInputNames.get(0))
            .addLayer("Joystick2",
                new ConvolutionLayer.Builder(4, 4).stride(2, 2).nOut(64).activation(Activation.RELU).build(),
                "Joystick1")
            .addLayer("Joystick3",
                new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(64).activation(Activation.RELU).build(),
                "Joystick2");

        String[] mergeInputs = new String[dependencyInputNames.size() + envInputNames.size()];

        for(int i = 0; i < dependencyInputNames.size(); i++){
            mergeInputs[i] = dependencyInputNames.get(i);
        }

        for(int i = 1; i < envInputNames.size(); i++){
            mergeInputs[i + dependencyInputNames.size() - 1] = envInputNames.get(i);
        }

        mergeInputs[mergeInputs.length - 1] = "Joystick3";

        builder
            .addVertex("JoystickMerge",
                new MergeVertex(),
                mergeInputs)
            .addLayer("Joystick4",
                new DenseLayer.Builder().nOut(512).activation(Activation.RELU).build(),
                "JoystickMerge")
            .addLayer("JoystickDirectionsInternal",
                new DenseLayer.Builder().nOut(9).activation(Activation.IDENTITY).build(),
                "Joystick4")
            .addLayer("JoystickDirections",
                new LossLayer.Builder().lossFunction(LossFunctions.LossFunction.SQUARED_LOSS).build(),
                "JoystickDirectionsInternal");

        this.outputNames = new ArrayList<>();
        outputNames.add("JoystickDirections");

        this.internalOutputNames = new ArrayList<>();
        internalOutputNames.add("JoystickDirectionsInternal");

        return internalOutputNames;
    }

    @Override
    public List<String> getOutputNames() {
        return outputNames;
    }

    @Override
    public List<String> getInternalOutputNames() {
        return internalOutputNames;
    }
}
