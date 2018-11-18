package drl;

import drl.agents.IAgent;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

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
        for(AgentDependencyGraph.Node node : this.getNodes()){
            ArrayList<Integer> inds = new ArrayList<>();
            for(int i = 0; i < outputs.length; i++){
                for(String name : node.agent.getOutputNames()){
                    if(name.equals(outputs[i])){
                        inds.add(i);
                    }
                }
            }

            agentInds.add(inds);
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
