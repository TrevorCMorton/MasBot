package drl.agents;

import drl.WeightedActivationRelu;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

public abstract class AbstractControlAgent implements IAgent{
    List<String> outputNames;
    String name;

    @Override
    public List<String> build(ComputationGraphConfiguration.GraphBuilder builder, List<String> envInputNames, List<String> dependencyInputNames, boolean buildOutputs, List<IActivation> activations) {
        String[] mergeInputs = new String[dependencyInputNames.size() + envInputNames.size()];

        for(int i = 0; i < dependencyInputNames.size(); i++){
            String inputName = dependencyInputNames.get(i) + "In";
            mergeInputs[i] = inputName;
        }

        for(int i = 0; i < envInputNames.size(); i++){
            mergeInputs[i + dependencyInputNames.size()] = envInputNames.get(i);
        }

        IActivation activation1;
        IActivation activation2;
        if(buildOutputs){
            activation1 = new ActivationReLU();
            activation2 = new ActivationReLU();
        }
        else{
            activation1 = new WeightedActivationRelu();
            activation2 = new WeightedActivationRelu();
        }
        activations.add(activation1);
        activations.add(activation2);

        builder.addLayer(this.name + this.getControlName() + "1",
                        new DenseLayer.Builder().nOut(1024).activation(activation1).build(),
                        mergeInputs)
                .addLayer(this.name + this.getControlName() + "2",
                        new DenseLayer.Builder().nOut(512).activation(activation2).build(),
                        this.name + this.getControlName() + "1");

        for(int i = 0; i < outputNames.size(); i++){
            String outputName = outputNames.get(i);


            if(buildOutputs){
                builder
                        .addLayer(outputName + "Internal",
                                new DenseLayer.Builder().nOut(1).weightInit(WeightInit.XAVIER).activation(Activation.IDENTITY).build(),
                                this.name + this.getControlName() + "2")
                        .addLayer(outputName,
                            new LossLayer.Builder().lossFunction(LossFunctions.LossFunction.L2).build(),
                            outputName + "Internal");
            }
            else{
                builder
                        .addLayer(outputName,
                                new DenseLayer.Builder().nOut(1).weightInit(WeightInit.XAVIER).activation(Activation.IDENTITY).build(),
                                this.name + this.getControlName() + "2");
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
