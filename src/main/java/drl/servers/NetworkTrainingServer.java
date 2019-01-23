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
    Socket socket;
    private Queue<Object> dataMessages;
    private boolean sendData;
    private boolean sendingData;
    private boolean run;

    public NetworkTrainingServer(String url){
        dataMessages = new LinkedList<>();
        sendData = true;
        sendingData = true;
        run = true;

        try {
            Socket tempSocket = new Socket(url, LocalTrainingServer.port);

            ObjectInputStream connectionIn = new ObjectInputStream(tempSocket.getInputStream());
            int port = (int)connectionIn.readObject();

            this.socket = new Socket(url, port);

            this.rawOutput = this.socket.getOutputStream();
            this.rawInput = this.socket.getInputStream();
            this.objectOutput = new ObjectOutputStream(rawOutput);
            this.objectInput = new ObjectInputStream(rawInput);
        }
        catch (Exception e){
            e.printStackTrace(System.out);
            System.out.println("Failed to initialze server connection");
        }
    }

    @Override
    public void addData(INDArray[] startState, INDArray endState, INDArray[] masks, float score, INDArray[] startLabels, INDArray[] endLabels) {
        dataMessages.add("addData");
        dataMessages.add(startState);
        dataMessages.add(endState);
        dataMessages.add(masks);
        dataMessages.add(score);
        dataMessages.add(startLabels);
        dataMessages.add(endLabels);
    }

    @Override
    public void addScore(double score) {
        try {
            sendData = false;
            while (sendingData) {
                Thread.sleep(5);
            }

            this.objectOutput.writeObject("addScore");
            this.objectOutput.writeObject(score);
        }
        catch(Exception e){
            System.out.println("IT GOOFED" + e);
        }
    }

    @Override
    public ComputationGraph getUpdatedNetwork() {
        try {
            sendData = false;
            while (sendingData) {
                Thread.sleep(5);
            }

            this.objectOutput.writeObject("getUpdatedNetwork");
            byte[] graphData = (byte[])this.objectInput.readObject();
            ByteArrayInputStream bais = new ByteArrayInputStream(graphData);
            ComputationGraph graph = ModelSerializer.restoreComputationGraph(bais, false);
            sendData = true;
            return graph;
        }
        catch(Exception e){
            System.out.println("IT GOOFED" + e);
            return null;
        }
    }

    @Override
    public AgentDependencyGraph getDependencyGraph() {
        try {
            sendData = false;
            while (sendingData) {
                Thread.sleep(5);
            }

            this.objectOutput.writeObject("getDependencyGraph");
            Object o = this.objectInput.readObject();
            AgentDependencyGraph dependencyGraph = (AgentDependencyGraph) o;
            sendData = true;
            return dependencyGraph;
        }
        catch(Exception e){
            System.out.println("IT GOOFED" + e);
            return null;
        }
    }

    @Override
    public void pause() {
        try {
            sendData = false;
            while (sendingData) {
                Thread.sleep(5);
            }
        }
        catch(Exception e){
            System.out.println("IT GOOFED" + e);
        }
    }

    @Override
    public void resume() {
        try {
            sendData = true;
            while (!sendingData) {
                Thread.sleep(5);
            }
        }
        catch(Exception e){
            System.out.println("IT GOOFED" + e);
        }
    }

    @Override
    public void stop() {
        try {
            this.flushQueue();
            this.objectOutput.writeObject("Go Kill Yourself");
            this.run = false;
            this.socket.close();
        }
        catch (Exception e){
            System.out.println(e);
        }
    }

    @Override
    public double getProb() {
        try {
            sendData = false;
            while (sendingData) {
                Thread.sleep(5);
            }

            this.objectOutput.writeObject("getProb");
            Object o = this.objectInput.readObject();
            double prob = (double) o;
            sendData = true;
            return prob;
        }
        catch(Exception e){
            System.out.println("Could Not Get Prob " + e);
            return 0;
        }
    }

    @Override
    public void run() {
        try {
            while (this.run) {
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
                    Thread.sleep(10);
                }
            }
        }
        catch(Exception e){
            System.out.println(e);
        }
    }

    private void flushQueue() throws Exception{
        sendData = false;
        while (sendingData) {
            Thread.sleep(5);
        }

        while(!dataMessages.isEmpty()) {
            this.objectOutput.writeObject(dataMessages.poll());
        }

        sendData = true;
    }
}
