package drl.servers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;

import static org.nd4j.linalg.ops.transforms.Transforms.*;

public class DataPoint {
    INDArray[] startState;
    INDArray[] endState;
    INDArray[] labels;
    INDArray[] masks;

    public DataPoint(INDArray[] startState, INDArray[] endState, INDArray[] labels, INDArray[] masks){
        this.startState = startState;
        this.endState = endState;
        this.labels = labels;
        this.masks = masks;
    }

    public MultiDataSet getDataSetWithQOffset(INDArray[] currentLabels, float decayRate){
        INDArray[] qOffsetLabels = new INDArray[this.labels.length];

        INDArray max = currentLabels[0];
        INDArray terminal = abs(this.labels[0]);
        for(int i = 1; i < currentLabels.length; i++){
            max = max.add(currentLabels[i]).add(abs(max.sub(currentLabels[i]))).mul(.5);
            terminal = terminal.add(abs(this.labels[i])).add(abs(terminal.sub(abs(this.labels[i])))).mul(.5);
        }

        terminal = not(greaterThanOrEqual(terminal, Nd4j.ones(terminal.shape())));

        for(int i = 0; i < this.labels.length; i++){
            qOffsetLabels[i] = this.labels[i].add(max.mul(terminal).mul(decayRate));
        }

        return new MultiDataSet(this.startState, qOffsetLabels, null, masks);
    }

    public INDArray[] getStartState() {
        return this.endState;
    }

    public INDArray[] getEndState() {
        return endState;
    }

    public INDArray[] getLabels() {
        return this.labels;
    }

    public INDArray[] getMasks() {
        return this.masks;
    }
}
