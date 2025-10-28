using BookRent.Catalog.Infrastructure.Interfaces;

namespace BookRent.Catalog.Api;

internal static class CatalogCustomerEndpoints
{
    internal static IResult GetAllBooks(HttpRequest request,
        IBookRepository repo)
    {
        
        return TypedResults.Ok("success");
    }
    
    internal static IResult GetBook(Guid id, HttpRequest request,  IBookRepository repo)
    {
        return Results.Ok();
    }
    
    internal static IResult SearchBook()
    {
        return Results.Ok();
    }
}