package drl.collections;

import drl.servers.DataPoint;

public class RankReplayer<T> implements IReplayer<T>{

    private Node root;
    private int maxSize;

    public RankReplayer(int maxSize){
        this.maxSize = maxSize;
    }

    @Override
    public void add(double error, T data) {
        System.out.println("add");
        this.root = addHelper(new Node(error, data), this.root);
        this.verifyHelper(this.root);

        if(this.size() > maxSize){
            System.out.println("remove last");
            this.root = removeLast(this.root);
            this.verifyHelper(this.root);
        }
    }

    private Node addHelper(Node toAdd, Node n){
        if(n == null){
            return toAdd;
        }
        else if(toAdd == null){
            return n;
        }

        if(n.key > toAdd.key){
            n.right = addHelper(toAdd, n.right);
            n.rightCount += toAdd.leftCount + toAdd.rightCount + 1;
        }
        else{
            n.left = addHelper(toAdd, n.left);
            n.leftCount += toAdd.leftCount + toAdd.rightCount + 1;
        }
        return n;
    }

    @Override
    public T get(int i) {
        if(i > this.size() - 1){
            System.out.println("Attempting access out of bounds, reducing index to " + (i - 1));
            return this.get(i - 1);
        }
        else {
            System.out.println("get");
            T temp = getHelper(i, this.root);
            this.verifyHelper(this.root);
            System.out.println("remove");
            this.root = remove(i, this.root);
            this.verifyHelper(this.root);
            return temp;
        }
    }

    private T getHelper(int i, Node n){
        if(n == null){
            System.out.println(i);
        }
        System.out.println(i + " " + n.leftCount + " " + n.rightCount);
        if(n.leftCount > i){
            return getHelper(i, n.left);
        }
        else if(n.leftCount == i){
            return n.data;
        }
        else{
            return getHelper(i - n.leftCount - 1, n.right);
        }
    }

    @Override
    public int size() {
        if(root == null){
            return 0;
        }
        return this.root.leftCount + this.root.rightCount + 1;
    }

    @Override
    public int getMaxSize() {
        return this.maxSize;
    }

    @Override
    public void prepopulate(T fillerData) {
        while(this.size() < this.maxSize){
            this.add(Math.random(), fillerData);
        }
    }

    private Node remove(int i, Node n){
        if(n.leftCount > i){
            Node temp = remove(i, n.left);
            n.left = temp;
            n.leftCount--;
            return n;
        }
        else if(n.leftCount == i){
            Node temp;
            if(n.leftCount > n.rightCount){
                temp = addHelper(n.right, n.left);
            }
            else{
                temp = addHelper(n.left, n.right);
            }

            return temp;
        }
        else{
            Node temp = remove(i - n.leftCount - 1, n.right);
            n.right = temp;
            n.rightCount--;
            return n;
        }
    }

    private Node removeLast(Node n){
        if(n.rightCount != 0){
            n.right = removeLast(n.right);
            n.rightCount--;
            return n;
        }
        else{
            return n.left;
        }
    }

    private boolean verify(){
        if(this.root == null){
            return true;
        }

        try{
            int trueCount = verifyHelper(this.root);

            if(this.size() != trueCount){
                return false;
            }
            if(this.size() > this.getMaxSize()){
                return false;
            }
        }
        catch (IndexOutOfBoundsException e){
            return false;
        }
        return true;
    }

    private int verifyHelper(Node n) throws IndexOutOfBoundsException{
        if(n == null){
            return 0;
        }
        int trueCount = verifyHelper(n.left) + verifyHelper(n.right) + 1;

        if(trueCount != n.leftCount + n.rightCount + 1){
            throw new IndexOutOfBoundsException(trueCount + " " + (n.leftCount + n.rightCount + 1));
        }

        return trueCount;
    }

    class Node{
        Node left;
        Node right;
        int leftCount;
        int rightCount;
        double key;
        T data;

        Node(double key, T data){
            this.left = null;
            this.right = null;
            this.leftCount = 0;
            this.rightCount = 0;
            this.key = key;
            this.data = data;
        }
    }
}
