package drl.agents;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

public class MeleeJoystickAgent extends AbstractControlAgent{
    public MeleeJoystickAgent(String name){
        this.name = name;

        this.outputNames = new ArrayList<>();

        String[] outputNameStubs = { "R", "N", "NE", "E", "SE", "S", "SW", "W", "NW" };
        for(String outputName : outputNameStubs){
            this.outputNames.add(name + outputName);
        }
    }

    @Override
    String getControlName() {
        return "Joystick";
    }
}
