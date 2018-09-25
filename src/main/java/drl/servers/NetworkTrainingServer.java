package drl.servers;

import drl.AgentDependencyGraph;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.*;
import java.net.Socket;
import java.rmi.RemoteException;
import java.util.LinkedList;
import java.util.Queue;

public class NetworkTrainingServer implements ITrainingServer{
    ObjectOutputStream objectOutput;
    OutputStream rawOutput;
    ObjectInputStream objectInput;
    InputStream rawInput;
    private Queue<Object> dataMessages;
    private boolean sendData;
    private boolean sendingData;

    public NetworkTrainingServer(String url){
        dataMessages = new LinkedList<>();
        sendData = true;
        sendingData = true;

        try {
            Socket socket = this.getServerSocket(url);

            this.rawOutput = socket.getOutputStream();
            this.rawInput = socket.getInputStream();
            this.objectOutput = new ObjectOutputStream(rawOutput);
            this.objectInput = new ObjectInputStream(rawInput);
        }
        catch (IOException e){
            System.out.println("Failed to initialze server connection");
        }
    }

    @Override
    public void addData(INDArray[] startState, INDArray[] endState, INDArray[] masks, float score) {
        dataMessages.add("addData");
        dataMessages.add(startState);
        dataMessages.add(endState);
        dataMessages.add(masks);
        dataMessages.add(score);
    }

    @Override
    public ComputationGraph getUpdatedNetwork() {
        try {
            sendData = false;
            while (sendingData) {
                Thread.sleep(10);
            }

            this.objectOutput.writeObject("getUpdatedNetwork");
            byte[] graphData = (byte[])this.objectInput.readObject();
            ByteArrayInputStream bais = new ByteArrayInputStream(graphData);
            ComputationGraph graph = ModelSerializer.restoreComputationGraph(bais, false);
            sendData = true;
            return graph;
        }
        catch(Exception e){
            System.out.println("IT GOOFED");
            return null;
        }
    }

    @Override
    public AgentDependencyGraph getDependencyGraph() {
        try {
            sendData = false;
            while (sendingData) {
                Thread.sleep(10);
            }

            this.objectOutput.writeObject("getDependencyGraph");
            Object o = this.objectInput.readObject();
            AgentDependencyGraph dependencyGraph = (AgentDependencyGraph) o;
            sendData = true;
            return dependencyGraph;
        }
        catch(Exception e){
            System.out.println("IT GOOFED");
            return null;
        }
    }

    @Override
    public void run() {
        try {
            while (true) {
                if(!sendData){
                    sendingData = false;
                    while(!sendData){
                        Thread.sleep(100);
                    }
                    sendingData = true;
                }

                if(!dataMessages.isEmpty()) {
                    if (dataMessages.peek() instanceof String) {
                        this.objectOutput.writeObject(dataMessages.poll());
                        while(!(dataMessages.peek() instanceof String) && !dataMessages.isEmpty()){
                            this.objectOutput.writeObject(dataMessages.poll());
                        }
                    }
                }
                else{
                    Thread.sleep(100);
                }
            }
        }
        catch(Exception e){
            System.out.println(e);
        }
    }

    public void flushQueue() throws Exception{
        sendData = false;
        while (sendingData) {
            Thread.sleep(10);
        }

        while(!dataMessages.isEmpty()) {
            this.objectOutput.writeObject(dataMessages.poll());
        }

        sendData = true;
    }

    private Socket getServerSocket(String url){
        boolean socketed = false;
        int portInd = 0;
        Socket socket = null;

        while(!socketed){
            try {
                socket = new Socket(url, LocalTrainingServer.ports[portInd]);
                socketed = true;
            }
            catch(IOException e){
                portInd++;
            }
        }

        return socket;
    }
}
