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

// 最终安全版本
void multiplication( NODE * h1, NODE * h2, NODE * h3){
    int terms[10000][2]; // [系数, 指数]
    int count = 0;
    
    NODE * p1 = h1->next;
    while(p1){
        NODE * p2 = h2->next;
        while(p2){
            if(count >= 10000) break; // 防止越界
            terms[count][0] = p1->coef * p2->coef;
            terms[count][1] = p1->exp + p2->exp;
            count++;
            p2 = p2->next;
        }
        p1 = p1->next;
    }
    
    if(count == 0){
        NODE * s = (NODE*)malloc(sizeof(NODE));
        s->coef = 0;
        s->exp = 0;
        s->next = NULL;
        h3->next = s;
        return;
    }
    
    // 冒泡排序
    for(int i = 0; i < count-1; i++){
        for(int j = 0; j < count-1-i; j++){
            if(terms[j][1] > terms[j+1][1]){
                int temp_coef = terms[j][0];
                int temp_exp = terms[j][1];
                terms[j][0] = terms[j+1][0];
                terms[j][1] = terms[j+1][1];
                terms[j+1][0] = temp_coef;
                terms[j+1][1] = temp_exp;
            }
        }
    }
    
    // 合并同类项 - 修复边界检查
    int write_idx = 0;
    for(int i = 1; i < count; i++){
        if(terms[i][1] == terms[write_idx][1]){
            terms[write_idx][0] += terms[i][0];
        } else {
            write_idx++;
            if(write_idx < 10000){ // 防止越界
                terms[write_idx][0] = terms[i][0];
                terms[write_idx][1] = terms[i][1];
            }
        }
    }
    count = write_idx + 1;
    
    // 构建结果链表
    NODE * tail = h3;
    for(int i = 0; i < count && i < 10000; i++){ // 双重边界检查
        if(terms[i][0] != 0){
            NODE * new_node = (NODE*)malloc(sizeof(NODE));
            if(new_node == NULL) break; // 防止内存分配失败
            new_node->coef = terms[i][0];
            new_node->exp = terms[i][1];
            new_node->next = NULL;
            tail->next = new_node;
            tail = new_node;
        }
    }
    
    // 如果结果为空，添加零多项式
    if(!h3->next){
        NODE * s = (NODE*)malloc(sizeof(NODE));
        if(s != NULL){
            s->coef = 0;
            s->exp = 0;
            s->next = NULL;
            h3->next = s;
        }
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