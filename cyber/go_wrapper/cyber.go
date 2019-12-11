package main



// /*
// #cgo CFLAGS: -I../ 
// #cgo LDFLAGS: -L../../bazel-bin/cyber/ -lcyber_core -lstdc++
// #include "cyber/cyber.h"
// #include "cyber/init.h"
// */
// import "C"
import "fmt"

func Init(module_name string){

}

type Node struct{
	node_name string
}


func NewNode(name string)(* Node){
	var node_ = new(Node)
	node_.node_name = name
	return node_
}
func (n Node) ShowNodeName() {
	fmt.Println(n.node_name)
}