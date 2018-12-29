package drl.servers;

import drl.MetaDecisionAgent;

public class GraphMetadata {
    int replaySize;
    int batchSize;
    float decayRate;
    int targetRotation;

    public GraphMetadata(int replaySize, int batchSize, float decayRate, int targetRotation){
        this.replaySize = replaySize;
        this.batchSize = batchSize;
        this.decayRate = decayRate;
        this.targetRotation = targetRotation;
    }

    public String getName(){
        StringBuilder sb = new StringBuilder();
        sb.append(this.replaySize);
        sb.append("-");
        sb.append(this.batchSize);
        sb.append("-");
        sb.append(this.decayRate);
        sb.append("-");
        sb.append(this.targetRotation);
        return sb.toString();
    }
}
