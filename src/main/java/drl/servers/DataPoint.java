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

    public MultiDataSet getDataSetWithQOffset(INDArray[] targetMaxs, float decayRate){
        INDArray[] qOffsetLabels = new INDArray[this.labels.length];

        INDArray absMax = abs(this.labels[0]);
        for(int i = 1; i < this.labels.length; i++){
            absMax = (absMax.add(abs(this.labels[i])).add(abs(absMax.sub(abs(this.labels[i]))))).mul(.5);
        }

        INDArray terminal = not(greaterThanOrEqual(absMax, Nd4j.ones(absMax.shape())));

        for(int i = 0; i < this.labels.length; i++){
            INDArray propReward = targetMaxs[i].mul(decayRate).mul(terminal);
            INDArray newLabel = this.labels[i].add(propReward);
            qOffsetLabels[i] = newLabel;
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
