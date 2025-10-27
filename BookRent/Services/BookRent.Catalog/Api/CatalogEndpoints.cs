using BookRent.Catalog.Infrastructure.Interfaces;

namespace BookRent.Catalog.Api;

public static class CatalogEndpoints
{
    public static void MapCatalogEndpoints(this IEndpointRouteBuilder app)
    {
        RouteGroupBuilder group = app.MapGroup("/catalog");

        group.MapGet("/books", GetAllBooks);
        group.MapGet("/books/search", SearchBook);
        group.MapGet("/book/{id:guid}", GetBook);
        group.MapPost("/addBooks", AddBooks);
        group.MapPost("/removeBooks", RemoveBooks);
    }

    private static IResult GetAllBooks(HttpRequest request,
                                        IBookRepository repo)
    {
        
        return TypedResults.Ok("success");
    }
    
    private static IResult GetBook(Guid id, HttpRequest request,  IBookRepository repo)
    {
        return Results.Ok();
    }
    private static IResult SearchBook()
    {
        return Results.Ok();
    }
    
    private static IResult AddBooks()
    {
        return Results.Ok();
    }
    
    private static IResult RemoveBooks()
    {
        return Results.Ok();
    }
    
    
}