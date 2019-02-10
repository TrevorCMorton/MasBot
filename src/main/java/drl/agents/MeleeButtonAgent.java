package drl.agents;

import java.util.ArrayList;

public class MeleeButtonAgent extends AbstractControlAgent{
    public MeleeButtonAgent(String name){
        this.name = name;

        this.outputNames = new ArrayList<>();

        String[] outputNameStubs = { "P", "R" };
        for(String outputName : outputNameStubs){
            this.outputNames.add(outputName + this.name);
        }
    }

    @Override
    String getControlName() {
        return "Button";
    }
}
