package agents;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;

import java.util.List;

public interface IAgent {

    List<String> build(ComputationGraphConfiguration.GraphBuilder builder, List<String> envInputNames, List<String> dependencyInputNames);

    List<String> getOutputNames();

    List<String> getInternalOutputNames();
}
