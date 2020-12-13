# Style



## Git

using .git to implement version control. 



## Naming

- class: `MyClass`

- object: `myObject`
- function: `myFunction`
- variable: `myVariable` , `my_variable` , `not_usually_used_value`



## singleton

For the objects that are frequently called but only created once in the beginning, we should try to use Singleton pattern to improve performance and reduce memory consumption. 

For example, the information of our devices, servers of devices, and others only created once at the start. 

We find that the cost of time for calling functions, calculation and compiling in our local computer is around several millisecond (ms) after using singleton, while hundreds of ms are required before using singleton. 





