package drl.agents;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

public abstract class AbstractControlAgent implements IAgent{
    List<String> outputNames;
    String name;

    @Override
    public List<String> build(ComputationGraphConfiguration.GraphBuilder builder, List<String> envInputNames, List<String> dependencyInputNames, boolean buildOutputs) {
        String[] mergeInputs = new String[dependencyInputNames.size() + envInputNames.size()];

        for(int i = 0; i < dependencyInputNames.size(); i++){
            String inputName = dependencyInputNames.get(i) + "In";
            mergeInputs[i] = inputName;
        }

        for(int i = 0; i < envInputNames.size(); i++){
            mergeInputs[i + dependencyInputNames.size()] = envInputNames.get(i);
        }

        builder.addLayer(this.name + this.getControlName() + "1",
                        new DenseLayer.Builder().nOut(512).activation(Activation.RELU).build(),
                        mergeInputs);

        for(int i = 0; i < outputNames.size(); i++){
            String outputName = outputNames.get(i);


            if(buildOutputs){
                builder
                        .addLayer(outputName + "Internal",
                                new DenseLayer.Builder().nOut(1).weightInit(WeightInit.XAVIER).activation(Activation.IDENTITY).build(),
                                this.name + this.getControlName() + "1")
                        .addLayer(outputName,
                            new LossLayer.Builder().lossFunction(LossFunctions.LossFunction.L2).build(),
                            outputName + "Internal");
            }
            else{
                builder
                        .addLayer(outputName,
                                new DenseLayer.Builder().nOut(1).weightInit(WeightInit.XAVIER).activation(Activation.IDENTITY).build(),
                                this.name + this.getControlName() + "1");
            }
        }

        return this.getOutputNames();
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

    @Override
    public String getName() {
        return this.name;
    }

    abstract String getControlName();
}
