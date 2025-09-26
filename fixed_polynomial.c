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

// 修复后的multiplication函数
void multiplication( NODE * h1, NODE * h2, NODE * h3){
    NODE * p1 = h1->next;
    
    while(p1 != NULL){
        NODE * p2 = h2->next;
        while(p2 != NULL){
            int coef = p1->coef * p2->coef;
            int exp = p1->exp + p2->exp;
            
            // 在h3中查找是否已存在相同指数的项
            NODE * prev = h3;
            NODE * curr = h3->next;
            
            while(curr != NULL && curr->exp < exp){
                prev = curr;
                curr = curr->next;
            }
            
            if(curr != NULL && curr->exp == exp){
                // 找到相同指数，累加系数
                curr->coef += coef;
                if(curr->coef == 0){
                    // 如果系数为0，删除该节点
                    prev->next = curr->next;
                    free(curr);
                }
            } else {
                // 没有找到相同指数，插入新节点
                NODE * new_node = (NODE*)malloc(sizeof(NODE));
                new_node->coef = coef;
                new_node->exp = exp;
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
    head3->next = NULL;  
    multiplication( head1, head2, head3 );  

    output( head3 );  
    return 0;  
}