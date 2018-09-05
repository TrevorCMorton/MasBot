import agents.IAgent;

import javax.xml.bind.ValidationException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

public class AgentDependencyGraph {
    private HashMap<String, Node> agentNodes;

    public AgentDependencyGraph(){
        this.agentNodes = new HashMap<>();
    }

    public void addAgent(String[] dependencies, IAgent agent, String agentName){
        Node agentNode = new Node(agent);

        if(dependencies != null){
            for(String dependency : dependencies){
                if(this.agentNodes.containsKey(dependency)){
                    Node dependencyNode = this.agentNodes.get(dependency);
                    dependencyNode.dependents.add(agentNode);
                    agentNode.dependencies.add(dependencyNode);
                }

            }
        }

        this.agentNodes.put(agentName, agentNode);
    }

    public Collection<Node> getNodes(){
        return this.agentNodes.values();
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
