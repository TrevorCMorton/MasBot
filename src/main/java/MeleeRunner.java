import drl.agents.IAgent;
import drl.agents.MeleeJoystickAgent;
import drl.AgentDependencyGraph;
import drl.MetaDecisionAgent;
import drl.servers.LocalTrainingServer;
import drl.servers.NetworkTrainingServer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import drl.servers.DummyTrainingServer;
import drl.servers.ITrainingServer;

import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;

public class MeleeRunner {

    public static void main(String[] args) throws Exception{
        //Nd4j.getMemoryManager().togglePeriodicGc(false);

        Runtime rt = Runtime.getRuntime();
        Process pr = rt.exec("dolphin-emu -e Melee.iso");

        /*
        AgentDependencyGraph dependencyGraph = new AgentDependencyGraph();

        IAgent joystickAgent = new MeleeJoystickAgent("M");
        IAgent cstickAgent = new MeleeJoystickAgent("C");
        dependencyGraph.addAgent(null, joystickAgent, "M");
        //dependencyGraph.addAgent(new String[]{"M"}, cstickAgent, "C");

        MetaDecisionAgent decisionAgent = new MetaDecisionAgent(dependencyGraph);

        //ITrainingServer server = new LocalTrainingServer(decisionAgent.getMetaGraph(), 10000, 128, .9f);
        ITrainingServer server = new DummyTrainingServer(decisionAgent.getMetaGraph());
        */

        //NetworkTrainingServer server = new NetworkTrainingServer("gauss.csse.rose-hulman.edu");
        ITrainingServer server = new NetworkTrainingServer("localhost");
        //ITrainingServer server = new NetworkTrainingServer("192.168.3.47");
        //ITrainingServer server = new NetworkTrainingServer("localhost");

        Thread t = new Thread(server);
        t.start();

        AgentDependencyGraph dependencyGraph = server.getDependencyGraph();
        MetaDecisionAgent decisionAgent = new MetaDecisionAgent(dependencyGraph, Double.parseDouble(args[0]), false);
        decisionAgent.setMetaGraph(server.getUpdatedNetwork());

        PythonBridge bridge = new PythonBridge();
        bridge.start();

        float[][] inputBuffer = new float[4][];

        for(int i = 0; i < inputBuffer.length; i++){
            inputBuffer[i] = new float[84 * 84];
        }

        INDArray[] prevActionMask = decisionAgent.getOutputMask(new String[0]);

        int[] shape = {1, 4, 84, 84};
        INDArray emptyFrame = Nd4j.zeros(shape);
        INDArray[] prevState = new INDArray[]{ emptyFrame, Nd4j.concat(1, prevActionMask) };

        long count = 0;

        while(true){
            long start = System.currentTimeMillis();

            if(count % 300 == 0){
                //server.flushQueue();

                System.out.println("Flushing took " + (System.currentTimeMillis() - start) + " ms");
            }

            if (bridge.isPostGame()){
                break;
                //ComputationGraph graph = server.getUpdatedNetwork();
                //decisionAgent.setMetaGraph(graph);
            }

            while(bridge.isPostGame()){
                Thread.sleep(10);
            }

            INDArray frame = getFrame(bridge, inputBuffer);

            INDArray[] state = new INDArray[]{ frame/*, Nd4j.concat(1, prevActionMask)*/  };

            String[] results = decisionAgent.eval(state);

            bridge.execute(results);

            float curScore = bridge.getReward();

            INDArray[] mask = decisionAgent.getOutputMask(results);

            long end = System.currentTimeMillis();
            if(end - start < 100) {
                if(curScore != 0) {
                    System.out.println(curScore);
                }
                server.addData(prevState, state, prevActionMask, curScore);
                Thread.sleep(100 - (end - start));
            }
            else{
                System.out.println((end - start)  + " ms ");
            }

            prevState = state;
            prevActionMask = mask;

            count++;
        }

        pr.destroy();
        //bridge.destroy();
        server.stop();
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
