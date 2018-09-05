import agents.IAgent;
import agents.MeleeJoystickAgent;
import org.jpy.PyObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

public class MeleeRunner {

    public static void main(String[] args) throws Exception{
        Runtime rt = Runtime.getRuntime();
        Process pr = rt.exec("dolphin-emu -e Melee.iso");

        AgentDependencyGraph dependencyGraph = new AgentDependencyGraph();

        IAgent joystickAgent = new MeleeJoystickAgent("M");
        IAgent cstickAgent = new MeleeJoystickAgent("C");
        dependencyGraph.addAgent(null, joystickAgent, "M");
        dependencyGraph.addAgent(new String[]{"M"}, cstickAgent, "C");

        MetaDecisionAgent decisionAgent = new MetaDecisionAgent(dependencyGraph);

        PythonBridge bridge = new PythonBridge();

        bridge.start();

        float[][] inputBuffer = new float[4][];

        for(int i = 0; i < inputBuffer.length; i++){
            inputBuffer[i] = new float[84 * 84];
        }

        while(true){
            long start = System.currentTimeMillis();

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

            INDArray[] inputs = new INDArray[]{ frame};

            String[] results = decisionAgent.eval(inputs);

            bridge.execute(results);

            long end = System.currentTimeMillis();
            System.out.println((end - start)  + "ms");

            if(end - start < 32){
                Thread.sleep(32 - (end - start));
            }
        }
    }
}
