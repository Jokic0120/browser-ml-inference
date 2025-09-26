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

// 基于你的原始代码，修复关键问题
void multiplication( NODE * h1, NODE * h2, NODE * h3){
    NODE * p1 = h1->next;
    while(p1){
        NODE * p2 = h2->next;
        while(p2){
            int co = p1->coef * p2->coef;
            int idx = p1->exp + p2->exp;
            
            // 关键修复：不要重置指针，而是在当前结果中查找插入位置
            NODE * pre = h3;
            NODE * p3 = h3->next;
            
            // 查找插入位置
            while(p3 && p3->exp < idx){
                pre = p3;
                p3 = p3->next;
            }
            
            if(p3 && p3->exp == idx){
                // 找到相同指数，累加系数
                p3->coef += co;
                if(p3->coef == 0){
                    // 删除系数为0的项
                    pre->next = p3->next;
                    free(p3);
                }
            } else {
                // 插入新项
                NODE * p = (NODE*)malloc(sizeof(NODE));
                p->coef = co;
                p->exp = idx;
                p->next = p3;
                pre->next = p;
            }
            
            p2 = p2->next;
        }
        p1 = p1->next;
    }
    
    // 如果结果为空，添加零多项式
    if(!h3->next){
        NODE * s = (NODE*)malloc(sizeof(NODE));
        s->coef = 0;
        s->exp = 0;
        s->next = NULL;
        h3->next = s;
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