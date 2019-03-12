package drl.agents;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.nd4j.linalg.activations.IActivation;

import java.io.Serializable;
import java.util.List;

public interface IAgent extends Serializable {

    List<String> build(ComputationGraphConfiguration.GraphBuilder builder, List<String> envInputNames, List<String> dependencyInputNames, boolean makeOutputs, List<IActivation> activation);

    List<String> getOutputNames();

    List<String> getInternalOutputNames();

    String getName();
}
