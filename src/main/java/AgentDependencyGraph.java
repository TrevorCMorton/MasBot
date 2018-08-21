import agents.IAgent;

import javax.xml.bind.ValidationException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class AgentDependencyGraph {
    private HashMap<String, Node> agentNodes;
    private List<Node> sources;

    public AgentDependencyGraph(){
        agentNodes = new HashMap<>();
        sources = new ArrayList<>();
    }

    public void addAgent(List<String> dependencies, IAgent agent, String agentName){
        Node agentNode = new Node(agent);

        if(dependencies != null){
            for(String dependency : dependencies){
                if(agentNodes.containsKey(dependency)){
                    Node dependencyNode = agentNodes.get(dependency);
                    dependencyNode.dependents.add(agentNode);
                    agentNode.dependencies.add(dependencyNode);
                }

            }
        }

        if(dependencies == null || dependencies.size() == 0){
            sources.add(agentNode);
        }

        agentNodes.put(agentName, agentNode);
    }

    public List<Node> getSources(){
        return this.sources;
    }

    public class Node{
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
