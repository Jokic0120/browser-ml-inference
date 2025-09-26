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

void multiplication( NODE * h1, NODE * h2, NODE * h3){
    NODE * p1 = h1->next;
    while(p1){
        NODE * p2 = h2->next;
        NODE * pre = h3;
        NODE * p3 = h3;
        if(h3->next) p3 = h3->next;
        while(p2){
            int co = p1->coef * p2->coef;
            int idx = p1->exp + p2->exp;
            if(!h3->next){
                NODE * p = ( NODE * ) malloc( sizeof(NODE) );
                p->coef = co;
                p->exp = idx;
                p->next = NULL;
                p3 = p;
                pre->next = p3;
                p2 = p2->next;
                continue;
            }
            while(p3){
                if(idx == p3->exp){
                    p3->coef += co;
                    if(p3->coef == 0){
                        pre->next = p3->next;
                        if(!p3->next) p3 = pre;
                        else p3 = pre->next;
                    }
                    break;
                }
                else if(idx < p3->exp){
                    NODE * p = ( NODE * ) malloc( sizeof(NODE) );
                    p->coef = co;
                    p->exp = idx;
                    pre->next = p;
                    pre = pre->next;
                    pre->next = p3;
                    break;
                }
                else if(!p3->next){
                    NODE * p = ( NODE * ) malloc( sizeof(NODE) );
                    p->coef = co;
                    p->exp = idx;
                    p->next = NULL;
                    if(pre != p3){
                        pre = pre->next;
                        p3 = p;
                        pre->next = p3;
                    }
                    else {
                        p3 = p;
                        pre->next=p3;
                    }
                    break;
                }
                pre = p3;
                p3 = p3->next;
            }
            p2 = p2->next;
        }
        p1 = p1->next;
    }
    if(!h3->next){
        NODE * s = ( NODE * ) malloc( sizeof(NODE) );
        s->coef = 0;
        s->exp = 0;
        h3->next = s;
        s->next = NULL;
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