.DATA
    a .WORD 10
    b .WORD 20
    sum .WORD 0
    t1 .WORD 0

.CODE
main:

    MOV a, #10
    MOV b, #20
    MOV R0, a
    ADD R0, b
    MOV t1, R0
    MOV R0, t1
    MOV sum, R0
    PUSH t1
    CALL print

    HALT

print:
    RET
