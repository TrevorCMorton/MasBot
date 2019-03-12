package drl;

import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

public class WeightedActivationRelu extends ActivationReLU {
    INDArray weight;

    public WeightedActivationRelu(){
        weight = null;
    }

    public void setWeight(INDArray weight) {
        this.weight = weight;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        Pair<INDArray, INDArray> result = super.backprop(in, epsilon);
        INDArray weightedDlDZ = result.getFirst().dup();
        Nd4j.getExecutioner().exec(new BroadcastMulOp(weightedDlDZ, this.weight, weightedDlDZ, 0), 0);
        this.weight = null;
        return new Pair<>(weightedDlDZ, result.getSecond());
    }
}
