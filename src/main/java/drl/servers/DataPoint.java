package drl.servers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;

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

        for(int i = 0; i < this.labels.length; i++){
            if(labels[i].amaxNumber().intValue() == 1){
                qOffsetLabels[i] = this.labels[i];
            }
            else {
                qOffsetLabels[i] = this.labels[i].add(currentLabels[i].mul(decayRate));
            }
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
