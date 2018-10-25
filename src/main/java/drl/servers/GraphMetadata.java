package drl.servers;

import drl.MetaDecisionAgent;

public class GraphMetadata {
    int replaySize;
    int batchSize;
    float decayRate;
    int commDepth;
    int targetRotation;

    public GraphMetadata(int replaySize, int batchSize, float decayRate, int commDepth, int targetRotation){
        this.replaySize = replaySize;
        this.batchSize = batchSize;
        this.decayRate = decayRate;
        this.commDepth = commDepth;
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
        sb.append("-");
        sb.append(this.commDepth);
        return sb.toString();
    }
}
