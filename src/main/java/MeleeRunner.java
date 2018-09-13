import agents.IAgent;
import agents.MeleeJoystickAgent;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jpy.PyObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import servers.ITrainingServer;
import servers.LocalTrainingServer;

import java.util.List;

public class MeleeRunner {

    public static void main(String[] args) throws Exception{
        Runtime rt = Runtime.getRuntime();
        Process pr = rt.exec("dolphin-emu -e Melee.iso");

        AgentDependencyGraph dependencyGraph = new AgentDependencyGraph();

        IAgent joystickAgent = new MeleeJoystickAgent("M");
        IAgent cstickAgent = new MeleeJoystickAgent("C");
        dependencyGraph.addAgent(null, joystickAgent, "M");
        //dependencyGraph.addAgent(new String[]{"M"}, cstickAgent, "C");

        MetaDecisionAgent decisionAgent = new MetaDecisionAgent(dependencyGraph);

        ITrainingServer server = new LocalTrainingServer(decisionAgent.getMetaGraph(), 10000, 2);

        PythonBridge bridge = new PythonBridge();

        bridge.start();

        Thread serverThread = new Thread(server);
        serverThread.start();

        float[][] inputBuffer = new float[4][];

        for(int i = 0; i < inputBuffer.length; i++){
            inputBuffer[i] = new float[84 * 84];
        }

        float prevScore = 0;
        INDArray[] prevActionMask = decisionAgent.getOutputMask(new String[0]);

        int[] shape = {1, 4, 84, 84};
        INDArray emptyFrame = Nd4j.zeros(shape);
        INDArray[] prevState = new INDArray[]{ emptyFrame, Nd4j.concat(1, prevActionMask) };

        long count = 0;

        while(true){
            long start = System.currentTimeMillis();

            if(count % 10000 == 0){
                //ComputationGraph graph = server.getUpdatedNetwork();
                //decisionAgent.setMetaGraph(graph);
            }

            INDArray frame = getFrame(bridge, inputBuffer);

            INDArray[] state = new INDArray[]{ frame/*, Nd4j.concat(1, prevActionMask)*/  };

            String[] results = decisionAgent.eval(state);

            bridge.execute(results);

            float curScore = bridge.getReward();

            INDArray[] mask = decisionAgent.getOutputMask(results);

            long end = System.currentTimeMillis();
            if(end - start < 100) {
                server.addData(prevState, state, prevActionMask, curScore - prevScore);
                Thread.sleep(100 - (end - start));
            }
            else{
                System.out.println((end - start)  + " ms " + server.getDataSize());
            }

            prevState = state;
            prevScore = curScore;
            prevActionMask = mask;

            count++;
        }
    }

    public static INDArray getFrame(PythonBridge bridge, float[][] inputBuffer){
        float[] inputFrame = bridge.getFlatFrame();
        for(int i = 0; i < inputBuffer.length; i++){
            float[] tempFrame = inputBuffer[i];
            inputBuffer[i] = inputFrame;
            inputFrame = tempFrame;
        }

        float[] flatFrame = new float[84 * 84 * 4];
        int pos = 0;
        for(int i = 0; i < inputBuffer.length; i++){
            for(int j = 0; j < inputBuffer[i].length; j++){
                flatFrame[pos] = inputBuffer[i][j];
                pos++;
            }
        }

        int[] shape = {1, 4, 84, 84};
        INDArray frame = Nd4j.create(flatFrame, shape, 'c');

        return frame;
    }
}
