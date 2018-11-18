public class MasterRunner {
    public static void main(String[] args) throws Exception{
        boolean sendData = true;
        String configPath = args[0];
        boolean setupCpu = true;
        int iterations = 0;
        int convergenceIterations = 1000000;
        while(true){
            float prob = iterations / convergenceIterations;
            String[] runnerArgs = {prob + "", sendData + "", configPath, setupCpu + ""};
            MeleeRunner.main(runnerArgs);
            iterations++;
        }
    }
}
