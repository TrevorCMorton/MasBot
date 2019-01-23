package drl;

import drl.agents.IAgent;

import java.io.Serializable;
import java.util.*;

public class AgentDependencyGraph implements Serializable {
    private HashMap<String, Node> agentNodes;
    private List<Node> roots;

    public AgentDependencyGraph(){
        this.agentNodes = new HashMap<>();
        this.roots = new ArrayList<>();
    }

    public void addAgent(String[] dependencies, IAgent agent, String agentName){
        Node agentNode = new Node(agent);

        if(dependencies != null && dependencies.length != 0){
            for(String dependency : dependencies){
                if(this.agentNodes.containsKey(dependency)){
                    Node dependencyNode = this.agentNodes.get(dependency);
                    dependencyNode.dependents.add(agentNode);
                    agentNode.dependencies.add(dependencyNode);
                }

            }
        }
        else{
            roots.add(agentNode);
        }

        this.agentNodes.put(agentName, agentNode);
    }

    public Collection<Node> getNodes(){
        return this.agentNodes.values();
    }

    public void resetNodes(){
        for(String nodeName : agentNodes.keySet()){
            Node node = agentNodes.get(nodeName);
            node.built = false;
        }
    }

    public ArrayList<ArrayList<Integer>> getAgentInds(String[] outputs){
        ArrayList<ArrayList<Integer>> agentInds = new ArrayList<>();
        Queue<Node> nodeList = new LinkedList<>();
        HashSet<Node> visited = new HashSet<>();

        for(Node node : this.getRoots()){
            nodeList.add(node);
        }

        while(!nodeList.isEmpty()){
            Node node = nodeList.poll();

            ArrayList<Integer> inds = new ArrayList<>();
            for(int i = 0; i < outputs.length; i++){
                for(String name : node.agent.getOutputNames()){
                    if(name.equals(outputs[i])){
                        inds.add(i);
                    }
                }
            }

            agentInds.add(inds);
            visited.add(node);
            for(Node dependent : node.dependents){
                boolean shouldAdd = true;
                for(Node dependency : dependent.dependencies){
                    if(!visited.contains(dependency)){
                        shouldAdd = false;
                    }
                }
                if(shouldAdd){
                    nodeList.add(dependent);
                }
            }
        }

        return agentInds;
    }

    public List<Node> getRoots(){
        return this.roots;
    }

    public class Node implements Serializable{
        public IAgent agent;
        public List<Node> dependents;
        public List<Node> dependencies;
        public boolean built;

        Node(IAgent agent){
            this.agent = agent;
            this.dependents = new ArrayList<>();
            this.dependencies = new ArrayList<>();
            this.built = false;
        }
    }
}
