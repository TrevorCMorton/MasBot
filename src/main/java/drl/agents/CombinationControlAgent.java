package drl.agents;

import java.util.ArrayList;

public class CombinationControlAgent extends AbstractControlAgent {

    public CombinationControlAgent(String[][] actionSets){
        this.name = "Full";

        this.outputNames = new ArrayList<>();

        ArrayList<String> actionCombos = new ArrayList<>();

        for(int j = 0; j < actionSets[0].length; j++){
            actionCombos.add(actionSets[0][j]);
        }

        for(int j = 1; j < actionSets.length; j++){
            ArrayList<String> subActionCombos = new ArrayList<>();

            for(int k = 0; k < actionCombos.size(); k++){
                for(int l = 0; l < actionSets[j].length; l++){
                    subActionCombos.add(actionCombos.get(k) + ":" + actionSets[j][l]);
                }
            }

            actionCombos = subActionCombos;
        }

        this.outputNames.addAll(actionCombos);
    }

    @Override
    String getControlName() {
        return "Combination";
    }
}
