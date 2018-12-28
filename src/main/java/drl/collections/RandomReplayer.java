package drl.collections;

import drl.servers.DataPoint;
import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.deeplearning4j.nn.graph.ComputationGraph;

public class RandomReplayer<T> implements IReplayer<T>{
    private CircularFifoQueue<T> points;


    public RandomReplayer(int maxSize){
        this.points = new CircularFifoQueue<>(maxSize);
    }

    @Override
    public void add(double error, T data) {
        this.points.add(data);
    }

    @Override
    public T get(int i) {
        T temp = this.points.get(i);
        this.points.remove(temp);
        return temp;
    }

    @Override
    public int size() {
        return this.points.size();
    }

    @Override
    public int getMaxSize() {
        return this.points.maxSize();
    }

    @Override
    public void prepopulate(T fillerData) {
        while(this.points.size() != this.points.maxSize()){
            this.points.add(fillerData);
        }
    }
}
