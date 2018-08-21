import agents.IAgent;
import agents.MeleeJoystickAgent;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class MeleeRunner {

    public static void main(String[] args) throws InterruptedException{
        AgentDependencyGraph dependencyGraph = new AgentDependencyGraph();

        IAgent joystickAgent = new MeleeJoystickAgent();
        dependencyGraph.addAgent(null, joystickAgent, "joystick");

        MetaDecisionAgent decisionAgent = new MetaDecisionAgent(dependencyGraph);

        while(true){
            INDArray emptyScreen = Nd4j.zeros(1, 4, 84, 84);
            INDArray[] mockInput = new INDArray[1];
            mockInput[0] = emptyScreen;

            long start = System.currentTimeMillis();
            decisionAgent.eval(mockInput);
            long end = System.currentTimeMillis();
            System.out.println((end - start)  + "ms");

            Thread.sleep(1000);
        }
    }
}
