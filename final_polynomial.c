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

// 修复后的高效算法
void multiplication( NODE * h1, NODE * h2, NODE * h3){
    // 使用动态数组存储乘积项
    int capacity = 1000;
    int *coeffs = (int*)malloc(capacity * sizeof(int));
    int *exps = (int*)malloc(capacity * sizeof(int));
    int count = 0;
    
    NODE * p1 = h1->next;
    while(p1){
        NODE * p2 = h2->next;
        while(p2){
            if(count >= capacity){
                capacity *= 2;
                coeffs = (int*)realloc(coeffs, capacity * sizeof(int));
                exps = (int*)realloc(exps, capacity * sizeof(int));
            }
            coeffs[count] = p1->coef * p2->coef;
            exps[count] = p1->exp + p2->exp;
            count++;
            p2 = p2->next;
        }
        p1 = p1->next;
    }
    
    if(count == 0){
        // 零多项式
        NODE * s = (NODE*)malloc(sizeof(NODE));
        s->coef = 0;
        s->exp = 0;
        s->next = NULL;
        h3->next = s;
        free(coeffs);
        free(exps);
        return;
    }
    
    // 简单冒泡排序
    for(int i = 0; i < count-1; i++){
        for(int j = 0; j < count-1-i; j++){
            if(exps[j] > exps[j+1]){
                int temp_coef = coeffs[j];
                int temp_exp = exps[j];
                coeffs[j] = coeffs[j+1];
                exps[j] = exps[j+1];
                coeffs[j+1] = temp_coef;
                exps[j+1] = temp_exp;
            }
        }
    }
    
    // 合并同类项
    int write_idx = 0;
    for(int i = 1; i < count; i++){
        if(exps[i] == exps[write_idx]){
            coeffs[write_idx] += coeffs[i];
        } else {
            write_idx++;
            coeffs[write_idx] = coeffs[i];
            exps[write_idx] = exps[i];
        }
    }
    count = write_idx + 1;
    
    // 构建结果链表
    NODE * tail = h3;
    for(int i = 0; i < count; i++){
        if(coeffs[i] != 0){
            NODE * new_node = (NODE*)malloc(sizeof(NODE));
            new_node->coef = coeffs[i];
            new_node->exp = exps[i];
            new_node->next = NULL;
            tail->next = new_node;
            tail = new_node;
        }
    }
    
    // 如果结果为空，添加零多项式
    if(!h3->next){
        NODE * s = (NODE*)malloc(sizeof(NODE));
        s->coef = 0;
        s->exp = 0;
        s->next = NULL;
        h3->next = s;
    }
    
    free(coeffs);
    free(exps);
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