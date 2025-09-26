#include <stdio.h>  
#include <stdlib.h>  
  
typedef struct node  
{   int    coef, exp;  
    struct node  *next;  
} NODE;  
  
void multiplication( NODE *, NODE * , NODE * );  
void input( NODE * );  
void output( NODE * );  
  
void input( NODE * head )  
{   int flag, sign, sum, x;  
    char c;  
  
    NODE * p = head;  
  
    while ( (c=getchar()) !='\n' )  
    {  
        if ( c == '<' )  
        {    sum = 0;  
             sign = 1;  
             flag = 1;  
        }  
        else if ( c =='-' )  
             sign = -1;  
        else if( c >='0'&& c <='9' )  
        {    sum = sum*10 + c - '0';  
        }  
        else if ( c == ',' )  
        {    if ( flag == 1 )  
             {    x = sign * sum;  
                  sum = 0;  
                  flag = 2;  
          sign = 1;  
             }  
        }  
        else if ( c == '>' )  
        {    p->next = ( NODE * ) malloc( sizeof(NODE) );  
             p->next->coef = x;  
             p->next->exp  = sign * sum;  
             p = p->next;  
             p->next = NULL;  
             flag = 0;  
        }  
    }  
}  
  
void output( NODE * head )  
{  
    while ( head->next != NULL )  
    {   head = head->next;  
        printf("<%d,%d>,", head->coef, head->exp );  
    }  
    printf("\n");  
}  

// 高效的多项式乘法实现
void multiplication( NODE * h1, NODE * h2, NODE * h3){
    NODE * p1 = h1->next;
    
    while(p1 != NULL){
        NODE * p2 = h2->next;
        while(p2 != NULL){
            int coef = p1->coef * p2->coef;
            int exp = p1->exp + p2->exp;
            
            // 简单插入到链表末尾，保持顺序
            NODE * new_node = (NODE*)malloc(sizeof(NODE));
            new_node->coef = coef;
            new_node->exp = exp;
            new_node->next = NULL;
            
            // 找到插入位置（按指数升序）
            NODE * prev = h3;
            NODE * curr = h3->next;
            
            while(curr != NULL && curr->exp < exp){
                prev = curr;
                curr = curr->next;
            }
            
            if(curr != NULL && curr->exp == exp){
                // 合并同类项
                curr->coef += coef;
                if(curr->coef == 0){
                    // 删除系数为0的项
                    prev->next = curr->next;
                    free(curr);
                }
                free(new_node); // 释放未使用的节点
            } else {
                // 插入新项
                new_node->next = curr;
                prev->next = new_node;
            }
            
            p2 = p2->next;
        }
        p1 = p1->next;
    }
    
    // 如果结果为空，添加零多项式
    if(h3->next == NULL){
        NODE * zero_node = (NODE*)malloc(sizeof(NODE));
        zero_node->coef = 0;
        zero_node->exp = 0;
        zero_node->next = NULL;
        h3->next = zero_node;
    }
}

int main()  
{   NODE * head1, * head2, * head3;  

    head1 = ( NODE * ) malloc( sizeof(NODE) );  
    input( head1 );  

    head2 = ( NODE * ) malloc( sizeof(NODE) );  
    input( head2 );  

    head3 = ( NODE * ) malloc( sizeof(NODE) );  
    h3->next = NULL;  
    multiplication( head1, head2, head3 );  

    output( head3 );  
    return 0;  
}