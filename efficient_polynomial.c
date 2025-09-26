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
    // 创建一个临时数组来存储所有乘积项
    int max_terms = 1000; // 假设最多1000个项
    int *coeffs = (int*)calloc(max_terms * 2, sizeof(int)); // 存储系数和指数
    int *exps = coeffs + max_terms;
    int count = 0;
    
    NODE * p1 = h1->next;
    while(p1 != NULL){
        NODE * p2 = h2->next;
        while(p2 != NULL){
            coeffs[count] = p1->coef * p2->coef;
            exps[count] = p1->exp + p2->exp;
            count++;
            p2 = p2->next;
        }
        p1 = p1->next;
    }
    
    // 按指数排序并合并同类项
    for(int i = 0; i < count; i++){
        for(int j = i + 1; j < count; j++){
            if(exps[i] > exps[j]){
                // 交换
                int temp_coef = coeffs[i];
                int temp_exp = exps[i];
                coeffs[i] = coeffs[j];
                exps[i] = exps[j];
                coeffs[j] = temp_coef;
                exps[j] = temp_exp;
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
        if(coeffs[i] != 0){ // 只添加非零项
            NODE * new_node = (NODE*)malloc(sizeof(NODE));
            new_node->coef = coeffs[i];
            new_node->exp = exps[i];
            new_node->next = NULL;
            tail->next = new_node;
            tail = new_node;
        }
    }
    
    // 如果结果为空，添加零多项式
    if(h3->next == NULL){
        NODE * zero_node = (NODE*)malloc(sizeof(NODE));
        zero_node->coef = 0;
        zero_node->exp = 0;
        zero_node->next = NULL;
        h3->next = zero_node;
    }
    
    free(coeffs);
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