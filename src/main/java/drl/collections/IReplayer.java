package drl.collections;

import drl.servers.DataPoint;
import org.deeplearning4j.nn.graph.ComputationGraph;

public interface IReplayer<T> {

    void add(double error, T data);
    T get(int i);
    int size();
    int getMaxSize();
    void prepopulate(T fillerData);
}
